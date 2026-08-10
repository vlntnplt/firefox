# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Build a markdown report of ML perftest results, split by ONNX backend.

The ML perf harness (toolkit/components/ml/tests/browser/head.js) runs each
test across every ONNX backend available on the machine and tags the metric
names accordingly, so a single job yields both a native and a wasm series when
the platform bundles libonnxruntime. It also emits an availability probe so a
missing series can be distinguished from a series that was never requested.

This tool collects those signals and renders:

  1. a build-configuration matrix -- where does libonnxruntime actually load
  2. a native-vs-wasm comparison per feature, wherever both ran
  3. an explicit list of features measured on only one backend, and why

Two input modes:

  --revision REV    pull job logs from Treeherder via treeherder-cli
  --log FILE ...    parse local `mach perftest` logs

Example:

    ./mach try perf --tasks-filter ml-perf
    ./mach python toolkit/components/ml/tests/tools/ml_perf_report.py \\
        --revision <rev> --output ml-perf-report.md
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# Emitted by browser_ml_ort_availability_probe.js, one per CI job.
PROBE_RE = re.compile(r"ML_ORT_PROBE\s+(\{.*?\})\s*$", re.MULTILINE)

# Emitted by reportMetrics() in head.js.
METRICS_RE = re.compile(r"perfMetrics \| (\[.*?\])\s*$", re.MULTILINE)

# Metric naming produced by perfTest(): "<FEATURE>-<TAG>-<metric>", where TAG
# is an uppercase backend tag, plus "<FEATURE>-native-available" for the probe.
BACKEND_TAGS = {"NATIVE": "onnx-native", "WASM": "onnx"}
TAGGED_RE = re.compile(
    rf"^(?P<feature>.+?)-(?P<tag>{'|'.join(BACKEND_TAGS)})-(?P<metric>.+)$"
)
AVAILABLE_SUFFIX = "-native-available"

# Metrics worth putting in the comparison table, in display order. Everything
# else is still collected but not tabulated, to keep the report readable.
HEADLINE_METRICS = [
    ("model-run-latency", "run", "ms"),
    ("initialization-latency", "init", "ms"),
    ("pipeline-ready-latency", "pipeline ready", "ms"),
    ("total-memory-usage", "memory", "MiB"),
    ("peak-memory-usage", "peak memory", "MiB"),
]


class Results:
    """Parsed perftest output, keyed by job label."""

    def __init__(self):
        # job -> list of probe dicts
        self.probes = defaultdict(list)
        # job -> feature -> backend tag -> metric -> median value
        self.metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        # job -> feature -> bool
        self.native_available = defaultdict(dict)

    def ingest(self, job, text):
        for match in PROBE_RE.finditer(text):
            try:
                self.probes[job].append(json.loads(match.group(1)))
            except json.JSONDecodeError:
                pass

        for match in METRICS_RE.finditer(text):
            try:
                entries = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            for entry in entries:
                self._ingest_metric(job, entry)

    def _ingest_metric(self, job, entry):
        name = entry.get("name", "")
        value = entry.get("value")
        if value is None:
            values = entry.get("values") or []
            value = statistics.median(values) if values else None
        if value is None:
            return

        if name.endswith(AVAILABLE_SUFFIX):
            feature = name[: -len(AVAILABLE_SUFFIX)]
            self.native_available[job][feature] = bool(value)
            return

        match = TAGGED_RE.match(name)
        if not match:
            # Untagged metric: a test that has not been migrated to the backend
            # matrix, or a hand-rolled metric. Record under a sentinel so the
            # report can flag it rather than silently dropping it.
            self.metrics[job]["(untagged)"]["-"][name] = value
            return

        self.metrics[job][match["feature"]][match["tag"]][match["metric"]] = value

    @property
    def jobs(self):
        return sorted(set(self.probes) | set(self.metrics))


def fetch_from_treeherder(revision, repo, filter_regex):
    """Download job logs for a try push and return {job_label: log_text}."""
    cmd = [
        "treeherder-cli",
        revision,
        "--repo",
        repo,
        "--json",
        "--fetch-logs",
        "--match-filter",
        "all",
    ]
    if filter_regex:
        cmd += ["--filter", filter_regex]

    print(f"$ {' '.join(cmd)}", file=sys.stderr)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        sys.exit(
            "treeherder-cli not found. Run `./mach bootstrap` to install it, "
            "or use --log to parse local perftest logs instead."
        )
    except subprocess.CalledProcessError as exc:
        sys.exit(f"treeherder-cli failed:\n{exc.stderr}")

    return extract_job_logs(proc.stdout)


def extract_job_logs(raw):
    """Pull {job_label: log_text} out of treeherder-cli --json output.

    The exact shape of that JSON is not pinned by this tool: we walk it looking
    for objects that carry both a job name and log text, so a schema change
    degrades to "found nothing" rather than a traceback.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        sys.exit("Could not parse treeherder-cli output as JSON.")

    logs = {}
    name_keys = ("job_type_name", "job_name", "name", "label")
    log_keys = ("log", "log_text", "logs", "content")

    def walk(node):
        if isinstance(node, dict):
            name = next(
                (node[k] for k in name_keys if isinstance(node.get(k), str)), None
            )
            text = next(
                (node[k] for k in log_keys if isinstance(node.get(k), str)), None
            )
            if name and text:
                platform = node.get("platform") or node.get("build_platform") or ""
                label = f"{platform} {name}".strip()
                logs[label] = logs.get(label, "") + "\n" + text
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(data)
    return logs


def ratio(native, wasm):
    if not native or not wasm:
        return None
    return wasm / native


def render(results):
    out = ["# ML perftest report: native ONNX vs WASM", ""]

    out += render_availability(results)
    out += render_comparisons(results)
    out += render_single_backend(results)

    return "\n".join(out).rstrip() + "\n"


def render_availability(results):
    out = ["## Native onnxruntime availability", ""]
    rows = []
    for job in results.jobs:
        for probe in results.probes[job]:
            rows.append((
                job,
                probe.get("platform", "?"),
                probe.get("arch", "?"),
                "yes" if probe.get("bundled") else "no",
                "yes" if probe.get("available") else "no",
                ""
                if probe.get("bundled") == probe.get("available")
                else "**MISMATCH**",
            ))

    if not rows:
        out += [
            "_No `ML_ORT_PROBE` lines found. Is "
            "`browser_ml_ort_availability_probe.js` running in this job set?_",
            "",
        ]
        return out

    out += [
        "| Job | Platform | Arch | Bundled | Loads | |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in sorted(set(rows)):
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    return out


def render_comparisons(results):
    out = ["## Backend comparison", ""]
    any_pair = False

    for job in results.jobs:
        job_lines = []
        for feature, by_tag in sorted(results.metrics[job].items()):
            if "NATIVE" not in by_tag or "WASM" not in by_tag:
                continue
            any_pair = True
            native, wasm = by_tag["NATIVE"], by_tag["WASM"]

            job_lines += [
                f"#### {feature}",
                "",
                "| Metric | native | wasm | wasm/native |",
                "| --- | ---: | ---: | ---: |",
            ]
            for key, label, unit in HEADLINE_METRICS:
                if key not in native and key not in wasm:
                    continue
                n, w = native.get(key), wasm.get(key)
                r = ratio(n, w)
                n_txt = f"{n:.1f}" if n is not None else "-"
                w_txt = f"{w:.1f}" if w is not None else "-"
                r_txt = f"{r:.2f}x" if r else "-"
                job_lines.append(f"| {label} ({unit}) | {n_txt} | {w_txt} | {r_txt} |")
            job_lines.append("")

        if job_lines:
            out += [f"### {job}", ""] + job_lines

    if not any_pair:
        out += [
            "_No feature ran on both backends. Either the native runtime is "
            "unavailable everywhere in this push (check the matrix above), or "
            "the tests have not been migrated to the backend matrix._",
            "",
        ]
    return out


def render_single_backend(results):
    out = ["## Measured on a single backend", ""]
    rows = []
    for job in results.jobs:
        for feature, by_tag in sorted(results.metrics[job].items()):
            if feature == "(untagged)":
                rows.append((
                    job,
                    feature,
                    "-",
                    "test not migrated to the backend matrix",
                ))
                continue
            if "NATIVE" in by_tag and "WASM" in by_tag:
                continue
            tag = ", ".join(sorted(by_tag))
            available = results.native_available[job].get(feature)
            if available is False:
                why = "native runtime unavailable on this configuration (expected)"
            elif available is True and "NATIVE" not in by_tag:
                why = "**native was available but not measured**"
            else:
                why = "no availability probe recorded"
            rows.append((job, feature, tag, why))

    if not rows:
        out += ["_None._", ""]
        return out

    out += ["| Job | Feature | Backend | Why |", "| --- | --- | --- | --- |"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--revision", help="Try push revision to pull logs from")
    source.add_argument(
        "--log", nargs="+", type=Path, help="Local perftest log file(s)"
    )
    parser.add_argument("--repo", default="try", help="Treeherder repo (default: try)")
    parser.add_argument(
        "--filter",
        dest="filter_regex",
        default="ml",
        help="Job name filter regex (default: ml)",
    )
    parser.add_argument(
        "--output", type=Path, help="Write markdown here instead of stdout"
    )
    args = parser.parse_args()

    results = Results()

    if args.revision:
        for job, text in fetch_from_treeherder(
            args.revision, args.repo, args.filter_regex
        ).items():
            results.ingest(job, text)
    else:
        for path in args.log:
            results.ingest(path.name, path.read_text(errors="replace"))

    report = render(results)

    if args.output:
        args.output.write_text(report)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(report)


if __name__ == "__main__":
    main()
