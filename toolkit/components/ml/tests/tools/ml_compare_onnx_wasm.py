# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Compare native ONNX and WASM ML perftest results from a try push.

The report opens with a coverage verdict -- every scheduled ML job must have
reported metrics, and every feature needs both a NATIVE and a WASM series
unless the variant is in DECLARED_SKIPS -- followed by the comparison tables.

Input modes:

    --revision REV    pull job logs from Treeherder via treeherder-cli
    --logs FILE ...   parse local `mach perftest` logs, or perfherder json

With --cache-dir, a second run for the same push reuses the download;
--refresh forces a refetch.

Example:

    ./mach try preset ml-perf
    ./mach python toolkit/components/ml/tests/tools/ml_compare_onnx_wasm.py \\
        --revision <rev> --output report.md
"""

import argparse
import json
import math
import re
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

# Emitted by reportMetrics() in head.js.
METRICS_RE = re.compile(r"perfMetrics \| (\[.*?\])\s*$", re.MULTILINE)

# Backend tags used in metric names, from BACKEND_TAGS in head.js. Non-ONNX
# tags are recognised but not compared.
ONNX_TAGS = {"NATIVE": "onnx-native", "WASM": "onnx"}
OTHER_TAGS = {"LLAMA.CPP": "llama.cpp"}
ALL_TAGS = {**ONNX_TAGS, **OTHER_TAGS}

# Task name suffixes of the per-backend variants, from the kind.yml split.
VARIANT_TAGS = {"-native": "NATIVE", "-wasm": "WASM"}

# Metric naming produced by runMLPerfTest(): "<FEATURE>-<metric>-<TAG>". The
# tag is a suffix so the names declared in perfMetadata stay substrings of the
# reported names; split_base() recovers the feature/metric boundary.
_TAG_ALTERNATION = "|".join(re.escape(tag) for tag in ALL_TAGS)
TAGGED_RE = re.compile(rf"^(?P<base>.+)-(?P<tag>{_TAG_ALTERNATION})$")

# Naming from logs that predate the suffix scheme: "<FEATURE>-<TAG>-<metric>",
# where the tag alternation delimits the feature name.
LEGACY_TAGGED_RE = re.compile(
    rf"^(?P<feature>.+?)-(?P<tag>{_TAG_ALTERNATION})-(?P<metric>.+)$"
)

# (family, tag) -> reason, for backend variants deliberately not scheduled.
# Mirror the reason as a comment next to the exclusion in the kind.yml.
DECLARED_SKIPS = {}

# family -> reason, for jobs whose perfMetadata sets `perfherder: false`.
NON_PERF_FAMILIES = {
    "ml-llama-smollm2-smoke": "functional smoke test, perfherder disabled",
}

# (label, unit, higher_is_better) for known metrics; describe() infers the
# rest from name suffixes.
METRIC_INFO = {
    "model-run-latency": ("run", "ms", False),
    "initialization-latency": ("init", "ms", False),
    "pipeline-ready-latency": ("pipeline ready", "ms", False),
    "e2e-run-latency": ("e2e run", "ms", False),
    "e2e-init-latency": ("e2e init", "ms", False),
    "concurrent-init-latency": ("concurrent init", "ms", False),
    "1st-token-latency": ("first token", "ms", False),
    "decoding-latency": ("decoding", "ms", False),
    "total-memory-usage": ("memory", "MiB", False),
    "peak-memory-usage": ("peak memory", "MiB", False),
    "decoding-tokenSpeed": ("decoding", "tokens/s", True),
    "decoding-charactersSpeed": ("decoding", "chars/s", True),
    "prompt-tokenSpeed": ("prompt", "tokens/s", True),
    "prompt-charactersSpeed": ("prompt", "chars/s", True),
}

# Columns of the headline table, as (candidate metrics, header, unit); the
# first candidate a feature reports wins.
SUMMARY_COLUMNS = [
    (("model-run-latency", "e2e-run-latency"), "run", "ms"),
    (("initialization-latency", "concurrent-init-latency"), "init", "ms"),
    (("total-memory-usage",), "memory", "MiB"),
]

# (feature, what this row measures) for raw metric feature names; rows
# sharing a feature are grouped in the tables. Keyed uppercase.
FEATURE_NAMES = {
    "AUTOFILL": ("Autofill", "single-engine pipeline"),
    "AUTOFILL-ENCODER": ("Autofill", "encoder alone"),
    "AUTOFILL-HEAD": ("Autofill", "classifier head alone"),
    "AUTOFILL-TWO-ENGINE": ("Autofill", "encoder + head together, per form"),
    "SMART-TAB-EMBEDDING": ("Smart tab groups", "embedding"),
    "SMART-TAB-TOPIC": ("Smart tab groups", "topic label"),
    "AGGLOMERATIVE": (
        "Smart tab groups",
        "engine start + embed + cluster 5 tabs, agglomerative",
    ),
    "LOGISTIC_REGRESSION_ANCHOR": (
        "Smart tab groups",
        "engine start + embed + regroup 5 tabs, logistic regression",
    ),
    "NEAREST_NEIGHBORS_ANCHOR": (
        "Smart tab groups",
        "engine start + embed + regroup 5 tabs, nearest neighbours",
    ),
    "INTENT": ("Suggest", "intent classification"),
    "NER": ("Suggest", "entity recognition"),
    "SUGGEST": ("Suggest", "end to end"),
    "INFERENCE": ("Suggest", "inference"),
    "INTENT-DEFAULT": ("Intent detection", "default model"),
    "INTENT-EN-FR": ("Intent detection", "en/fr model"),
    "PDFJS-ALT-TEXT": ("PDF.js alt text", ""),
    "SHORT-SEMANTIC": ("Semantic history search", "short query"),
    "LONG-SEMANTIC": ("Semantic history search", "long query"),
    "LONG-MULTIPLE-SEMANTIC": ("Semantic history search", "long query, many results"),
}

# Features whose number does not mean what its row suggests; printed under
# the table that shows them.
FEATURE_CAVEATS = {
    "AGGLOMERATIVE": (
        "The smart tab clustering rows time a whole `generateClusters()` / "
        "`findNearestNeighbors()` call, and the stubbed `_generateEmbeddings()` "
        "inside it creates the engine, so most of the number is the model load "
        "-- their native gain tracks the embedding *init* ratio, not its run "
        "ratio. They are also a single un-warmed sample "
        "(`browser_ml_smart_tab_clustering_perf.js`, N_TABS loop). Treat them "
        "as cold-start numbers, not clustering cost."
    ),
}
FEATURE_CAVEATS["LOGISTIC_REGRESSION_ANCHOR"] = FEATURE_CAVEATS["AGGLOMERATIVE"]
FEATURE_CAVEATS["NEAREST_NEIGHBORS_ANCHOR"] = FEATURE_CAVEATS["AGGLOMERATIVE"]

# Human names for Treeherder platform slugs; unknown slugs fall through
# unchanged.
PLATFORM_NAMES = {
    "linux1804-64-shippable": "Linux 18.04 x86-64",
    "linux2404-64-shippable": "Linux 24.04 x86-64",
    "macosx1015-64-shippable-qr": "macOS 10.15 x86-64",
    "windows11-64-24h2-shippable": "Windows 11 24H2 x86-64",
    "windows11-64-24h2-hw-ref-shippable": "Windows 11 24H2 reference hardware",
}

# Tasks measuring the ML plumbing rather than a shipping feature; their
# series are excluded from the comparison unless --include-harness-tests.
# Their jobs still count for the coverage verdict.
HARNESS_FAMILIES = {
    "ml-perf": "engine template test, measures a stand-in model",
    "ml-multi-perf": "multi-engine stress test, measures numbered test engines",
}

# Setup-phase latencies, excluded when falling back to "any latency" for
# features that do not report model-run-latency.
NOT_A_RUN_METRIC = ("initialization", "pipeline-ready", "e2e-init", "cold-start")


# Feature names that only appear inside metric names, never as a row of
# their own in FEATURE_NAMES.
EXTRA_SPLIT_FEATURES = ("SINGLE-TAB", "SEMANTIC")

_METRICS_BY_LENGTH = sorted(METRIC_INFO, key=len, reverse=True)
_FEATURES_BY_LENGTH = sorted(
    {*FEATURE_NAMES, *EXTRA_SPLIT_FEATURES}, key=len, reverse=True
)


def split_base(base):
    """(feature, metric) for a tag-stripped subtest name.

    The boundary is not marked in the name, so try the known metric names as
    a suffix first, then the known feature names as a prefix. A name matching
    neither is its own feature, with itself as the only metric.
    """
    for known in _METRICS_BY_LENGTH:
        for metric in (f"cold-start-{known}", known):
            if base.endswith(f"-{metric}"):
                return base[: -len(metric) - 1], metric
    upper = base.upper()
    for feature in _FEATURES_BY_LENGTH:
        if upper.startswith(f"{feature}-"):
            return base[: len(feature)], base[len(feature) + 1 :]
    return base, base


def parse_tagged(name):
    """(feature, tag, metric) for a backend-tagged subtest name, else None."""
    match = TAGGED_RE.match(name)
    if match:
        feature, metric = split_base(match["base"])
        return feature, match["tag"], metric
    match = LEGACY_TAGGED_RE.match(name)
    if match:
        return match["feature"], match["tag"], match["metric"]
    return None


def describe(metric):
    """(label, unit, higher_is_better) for a metric name."""
    if metric in METRIC_INFO:
        return METRIC_INFO[metric]
    base = metric[len("cold-start-") :] if metric.startswith("cold-start-") else metric
    if base in METRIC_INFO:
        label, unit, higher = METRIC_INFO[base]
        return (f"cold start {label}", unit, higher)
    if metric.endswith("Speed"):
        return (metric, "/s", True)
    if "memory" in metric:
        return (metric, "MiB", False)
    if metric.endswith("latency"):
        return (metric, "ms", False)
    return (metric, "", False)


def gain(native, wasm, higher_is_better=False):
    """How much better native is than wasm. Above 1.0 favours native."""
    if not native or not wasm:
        return None
    return native / wasm if higher_is_better else wasm / native


def geomean(values):
    values = [v for v in values if v and v > 0]
    if not values:
        return None
    return math.exp(statistics.fmean(math.log(v) for v in values))


def platform_name(platform):
    return PLATFORM_NAMES.get(platform, platform) or "local logs"


def canonical_task(task):
    """The task name with the platform prefix dropped, e.g. "ml-perf-autofill".

    The same test is called "linux-ml-perf-smart-tab" on one platform and
    "windows11-24h2-ref-ml-perf-smart-tab" on another. Comparing coverage
    across platforms needs the one name they share.
    """
    index = task.find("ml-")
    return task[index:] if index != -1 else task


def split_variant(task):
    """("ml-perf-autofill", "NATIVE") for a canonical "ml-perf-autofill-native".

    Variant None means the task is not backend-split: either it drives a
    non-ONNX backend (llama.cpp), or the logs predate the split and one job
    carries both series.
    """
    for suffix, tag in VARIANT_TAGS.items():
        if task.endswith(suffix):
            return task[: -len(suffix)], tag
    return task, None


def split_label(label):
    """Split "<platform> <task>" and drop the platform echoed in the task name.

    Task names repeat the platform and the build type ("...-ml-perf-autofill-
    linux1804-64-shippable/opt"), which is pure width in a table that already
    has a platform heading.
    """
    platform, _, task = label.partition(" ")
    if not task:
        # A local log file: no platform to speak of, the file name is the task.
        return "", platform
    task = task.partition("/")[0]
    task = re.sub(r"^perftest-", "", task)
    if platform and task.endswith(f"-{platform}"):
        task = task[: -len(platform) - 1]
    return platform, task


class Job:
    """One CI job (or one local log file) and everything it measured."""

    def __init__(self, label):
        self.label = label
        self.platform, self.task = split_label(label)
        self.family, self.variant = split_variant(canonical_task(self.task))
        # feature -> tag -> metric -> median value
        self.metrics = defaultdict(lambda: defaultdict(dict))
        # metric name -> value, for names that carry no backend tag
        self.untagged = {}


class Results:
    def __init__(self, include_harness=False):
        self.jobs = {}
        self.include_harness = include_harness
        # (platform, family, variant) -> status key, for every job in the
        # push, whether or not it produced anything.
        self.statuses = {}

    def record_status(self, platform, task, state, result):
        if state != "completed":
            status = state if state in ("pending", "running") else "pending"
        elif result == "data":
            status = "data"
        elif result == "success":
            status = "no data"
        else:
            status = "failed"
        family, variant = split_variant(canonical_task(task))
        key = (platform, family, variant)
        # "data" is the final word: a job can report metrics and still be
        # revisited by the caller with only its treeherder state.
        if self.statuses.get(key) != "data":
            self.statuses[key] = status

    def job(self, label):
        return self.jobs.setdefault(label, Job(label))

    def ingest_perfherder(self, label, data):
        """Ingest one perfherder-data-*.json.log artifact.

        Returns the number of subtests recorded, so the caller can tell a job
        with no ML data from one it never managed to read.
        """
        if data.get("framework", {}).get("name") == "build_metrics":
            return 0
        count = 0
        for suite in data.get("suites", []):
            for subtest in suite.get("subtests", []):
                self._ingest_metric(label, subtest)
                count += 1
        return count

    def ingest_log(self, label, text):
        for match in METRICS_RE.finditer(text):
            try:
                entries = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            for entry in entries:
                self._ingest_metric(label, entry)

    def _ingest_metric(self, label, entry):
        name = entry.get("name", "")
        value = entry.get("value")
        if value is None:
            # "values" comes from head.js reportMetrics(), "replicates" from a
            # perfherder subtest.
            values = entry.get("values") or entry.get("replicates") or []
            value = statistics.median(values) if values else None
        if value is None:
            return

        # Soft-failure metrics from before the per-backend job split.
        if name.lower().endswith("-native-available"):
            return

        job = self.job(label)
        parsed = parse_tagged(name)
        if not parsed:
            job.untagged[name] = value
            return
        feature, tag, metric = parsed
        if metric == "backend-error":
            return
        job.metrics[feature][tag][metric] = value

    def reported_jobs(self):
        return sorted(self.jobs.values(), key=lambda j: (j.platform, j.task))

    def platforms(self):
        return sorted({job.platform for job in self.jobs.values()})

    def features(self, platform, include_harness=None):
        """Merged feature -> tag -> metrics for a platform, plus a sources map
        from feature to reporting task families. The NATIVE and WASM series of
        a feature come from two different jobs.
        """
        if include_harness is None:
            include_harness = self.include_harness
        features = defaultdict(lambda: defaultdict(dict))
        sources = defaultdict(set)
        # Feature name casing is inconsistent across tests; merge
        # case-insensitively, first spelling wins.
        canon = {}
        for job in self.reported_jobs():
            if job.platform != platform:
                continue
            if not include_harness and job.family in HARNESS_FAMILIES:
                continue
            for feature, by_tag in job.metrics.items():
                key = canon.setdefault(feature.upper(), feature)
                for tag, metrics in by_tag.items():
                    features[key][tag].update(metrics)
                sources[key].add(job.family)
        return features, sources

    def compared_features(self):
        """Every (platform, feature, native, wasm) with both ONNX series."""
        for platform in self.platforms():
            features, _ = self.features(platform)
            for feature in sorted(features, key=str.upper):
                native = features[feature].get("NATIVE", {})
                wasm = features[feature].get("WASM", {})
                if native and wasm:
                    yield platform, feature, native, wasm


def column_metric(column, native, wasm):
    """Which metric fills a headline column for this feature, if any."""
    candidates, header, _ = column
    keys = set(native) | set(wasm)
    for candidate in candidates:
        if candidate in keys:
            return candidate
    if header != "run":
        return None
    # A test that measures something other than one model run -- a whole
    # clustering pass, say -- still has a "how long did it take" number.
    others = sorted(
        k for k in keys if k.endswith("latency") and not k.startswith(NOT_A_RUN_METRIC)
    )
    return others[0] if others else None


def fetch_from_treeherder(revision, repo, filter_regex, cache_dir, refresh):
    """Download job logs for a push, or reuse a previous download.

    Returns the treeherder-cli JSON, which names the log directory of every job
    and is the only place a job that produced nothing at all is visible.
    """
    cached = cache_dir / "treeherder.json"
    if cached.exists() and not refresh:
        print(f"Reusing {cached} (pass --refresh to refetch)", file=sys.stderr)
        return json.loads(cached.read_text())

    cmd = [
        "treeherder-cli",
        revision,
        "--repo",
        repo,
        "--json",
        "--fetch-logs",
        "--match-filter",
        "all",
        # Without this, logs land in a temp directory that treeherder-cli
        # removes on exit, and every log_dir it reports is already gone by the
        # time we read the JSON.
        "--cache-dir",
        str(cache_dir),
    ]
    if filter_regex:
        # "--filter=X" rather than "--filter X": the default filter starts with
        # a dash, which the CLI would otherwise read as another option.
        cmd.append(f"--filter={filter_regex}")

    print(f"$ {' '.join(cmd)}", file=sys.stderr)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        sys.exit(
            "treeherder-cli not found. Run `./mach bootstrap` to install it, "
            "or use --logs to parse local perftest logs instead."
        )
    except subprocess.CalledProcessError as exc:
        sys.exit(f"treeherder-cli failed:\n{exc.stderr}")

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        sys.exit("Could not parse treeherder-cli output as JSON.")

    cached.write_text(json.dumps(data))
    return data


def ingest_treeherder(results, data):
    """Feed every job's downloaded artifacts into `results`.

    Also records the state of every job in the push, run or not. Failing
    jobs' logs are not ingested: a failed job aborted mid-measurement.
    """
    for entry in data.get("jobs", []):
        job = entry.get("job", {})
        label = " ".join(
            part
            for part in (job.get("platform"), job.get("job_type_name"))
            if isinstance(part, str)
        ).strip() or str(job.get("id", "unknown"))
        platform, task = split_label(label)

        found = False
        if job.get("result") in ("success", "data"):
            log_dir = entry.get("log_dir")
            for path in sorted(Path(log_dir).glob("*")) if log_dir else []:
                if not path.is_file() or not path.name.startswith("perfherder-data-"):
                    continue
                try:
                    payload = json.loads(path.read_text())
                except json.JSONDecodeError:
                    continue
                found |= results.ingest_perfherder(label, payload) > 0

        results.record_status(platform, task, job.get("state"), job.get("result"))
        if found:
            results.record_status(platform, task, "completed", "data")
        else:
            results.jobs.pop(label, None)


def fmt(value):
    if value is None:
        return "-"
    return f"{value:,.1f}" if abs(value) < 10000 else f"{value:,.0f}"


# Below this, native lost by more than run-to-run noise would explain.
REGRESSION_THRESHOLD = 0.95


def fmt_wasm(value, ratio):
    """The wasm cell: its value, and how it compares to native."""
    if value is None:
        return "-"
    if not ratio:
        return fmt(value)
    verdict = f"{ratio:.1f}x worse" if ratio >= 1 else f"{1 / ratio:.1f}x better"
    if ratio < REGRESSION_THRESHOLD:
        verdict = f"**{verdict}**"
    return f"{fmt(value)} ({verdict})"


def table(header, rows, align=None):
    if not rows:
        return []
    align = align or ["---"] * len(header)
    return (
        ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
        + ["| " + " | ".join(row) + " |" for row in rows]
        + [""]
    )


def feature_name(feature, run_metric=None):
    """(feature, what this row measures) for a raw metric feature name."""
    known = FEATURE_NAMES.get(feature.upper())
    if known:
        return known
    # No human name for it, so show the raw one -- and say which metric the run
    # column is showing, since it is not the usual one.
    if run_metric and run_metric not in SUMMARY_COLUMNS[0][0]:
        return (feature, f"<sup>{run_metric}</sup>")
    return (feature, "")


def flat_name(feature):
    """The same thing on one line, for prose and headings."""
    name, detail = feature_name(feature)
    return f"{name}: {detail}" if detail else name


def variant_label(family, variant):
    return f"{family}{'-' + variant.lower() if variant else ''}"


def coverage_problems(results):
    """Everything that violates the report's two rules, one line each."""
    problems = []

    for (platform, family, variant), status in sorted(results.statuses.items()):
        task = variant_label(family, variant)
        where = platform_name(platform)
        if status == "failed":
            problems.append(
                f"- **`{task}` failed on {where}.** Its backend has no numbers "
                "in this push; the job log has the error."
            )
        elif status == "no data" and family not in NON_PERF_FAMILIES:
            problems.append(
                f"- **`{task}` ran green on {where} but reported no ML "
                "metrics.** A green job that measured nothing is a broken "
                "job, not a passing one."
            )

    # Rule 2a, scheduling: a family that is backend-split must have both
    # variants wherever it has one.
    families = defaultdict(set)
    for platform, family, variant in results.statuses:
        if variant:
            families[(platform, family)].add(variant)
    for (platform, family), variants in sorted(families.items()):
        for missing in sorted(set(VARIANT_TAGS.values()) - variants):
            skip = DECLARED_SKIPS.get((family, missing))
            if skip:
                continue
            problems.append(
                f"- **`{variant_label(family, missing)}` is not scheduled on "
                f"{platform_name(platform)}** while its counterpart is, and "
                "no DECLARED_SKIPS entry covers it."
            )

    # Rule 2b, data: both series must exist for every feature. Suppressed
    # for a (platform, family) already reported above.
    reported = {
        (platform, family)
        for (platform, family, _), status in results.statuses.items()
        if status in ("failed", "no data", "pending", "running")
    }
    reported |= {(platform, family) for (platform, family), _ in families.items()}
    for platform in results.platforms():
        features, sources = results.features(platform, include_harness=True)
        for feature in sorted(features, key=str.upper):
            tags = set(features[feature]) & set(ONNX_TAGS)
            if not tags or len(tags) == 2:
                continue
            if any((platform, family) in reported for family in sources[feature]):
                continue
            missing = (set(ONNX_TAGS) - tags).pop()
            if any(
                DECLARED_SKIPS.get((family, missing)) for family in sources[feature]
            ):
                continue
            problems.append(
                f"- **`{flat_name(feature)}` has no {missing.lower()} series "
                f"on {platform_name(platform)}** (reported by "
                + ", ".join(f"`{s}`" for s in sorted(sources[feature]))
                + "), and no job-level problem explains it."
            )

    return problems


def render_coverage(results):
    out = ["## Coverage", ""]

    if not results.statuses:
        out += [
            "Local logs: no scheduling information, so the verdict below only "
            "covers series pairing in the files given.",
            "",
        ]

    pending = sorted(
        (platform, family, variant)
        for (platform, family, variant), status in results.statuses.items()
        if status in ("pending", "running")
    )
    problems = coverage_problems(results)

    ok = sum(1 for status in results.statuses.values() if status == "data")
    if not problems:
        out.append(
            f"**All {ok} completed ML jobs reported data and every native/wasm "
            "series pair is complete.**"
            if results.statuses
            else "**Every feature in the given logs has both series.**"
        )
    else:
        out.append(
            f"**{len(problems)} coverage problem"
            f"{'s' if len(problems) > 1 else ''}** "
            f"({ok} jobs reported data):"
        )
        out.append("")
        out += problems

    if pending:
        out += [
            "",
            f"{len(pending)} job{'s' if len(pending) > 1 else ''} not finished "
            "yet: "
            + ", ".join(
                f"`{variant_label(family, variant)}` ({platform_name(platform)})"
                for platform, family, variant in pending
            )
            + ". The numbers below cover only what has reported.",
        ]

    for (family, tag), reason in sorted(DECLARED_SKIPS.items()):
        out.append(f"- Declared skip: `{variant_label(family, tag)}` -- {reason}")

    out.append("")
    return out


def render_summary(results):
    out = ["## Summary", ""]

    pairs = list(results.compared_features())
    if not pairs:
        out += [
            "No feature has both ONNX series in this push -- see the coverage "
            "verdict above.",
            "",
        ]
        return out

    platforms = {platform for platform, _, _, _ in pairs}
    out.append(
        f"- {len(pairs)} features measured on both backends, on "
        + ", ".join(sorted(platform_name(platform) for platform in platforms))
        + "."
    )

    for column in SUMMARY_COLUMNS:
        _, label, unit = column
        gains = []
        for _, _, native, wasm in pairs:
            key = column_metric(column, native, wasm)
            if key:
                gains.append(gain(native.get(key), wasm.get(key), describe(key)[2]))
        value = geomean([g for g in gains if g])
        if value:
            direction = "faster" if unit == "ms" else "smaller"
            out.append(
                f"- **{label}: native is {value:.2f}x {direction}** "
                f"(geomean over {len([g for g in gains if g])} features)."
            )

    out.append("")
    return out


def render_comparison(results):
    """One table per platform: the headline numbers, biggest win first."""
    out = []
    for platform in results.platforms():
        features, _ = results.features(platform)
        # feature -> [(run gain, detail, cells)]: one feature's rows stay
        # together when the table is sorted by gain.
        families = defaultdict(list)
        caveats = []
        for feature in sorted(features, key=str.upper):
            native = features[feature].get("NATIVE", {})
            wasm = features[feature].get("WASM", {})
            if not (native and wasm):
                continue
            run_metric = column_metric(SUMMARY_COLUMNS[0], native, wasm)
            name, detail = feature_name(feature, run_metric)
            cells = []
            run_gain = 0.0
            for column in SUMMARY_COLUMNS:
                key = column_metric(column, native, wasm)
                n = native.get(key) if key else None
                w = wasm.get(key) if key else None
                g = gain(n, w, describe(key)[2] if key else False)
                cells += [fmt(n), fmt_wasm(w, g)]
                if column is SUMMARY_COLUMNS[0]:
                    run_gain = g or 0.0
            families[name].append((run_gain, detail, cells))
            caveat = FEATURE_CAVEATS.get(feature.upper())
            if caveat and caveat not in caveats:
                caveats.append(caveat)

        if not families:
            continue

        rows = []
        for name in sorted(families, key=lambda f: -max(g for g, _, _ in families[f])):
            for i, (_, detail, cells) in enumerate(
                sorted(families[name], reverse=True)
            ):
                # Only the first row of a feature group is labelled.
                rows.append(["**" + name + "**" if not i else "", detail] + cells)

        header = ["Feature", "Measures"]
        align = ["---", "---"]
        for _, label, unit in SUMMARY_COLUMNS:
            header += [f"native {label} ({unit})", f"wasm {label}"]
            align += ["---:", "---:"]

        out += [f"## Native vs wasm on {platform_name(platform)}", ""]
        out += table(header, rows, align)
        out += [
            "Each wasm cell says how that number compares to native, so "
            '"9.6x worse" means wasm took 9.6 times as long, and bold marks '
            "the cases where native lost. A dash means the test does not "
            "report that metric: memory is a whole-process number, so a test "
            "that times two models separately can only attribute it to the "
            "run that has both of them loaded.",
            "",
        ]
        out += [f"Caveat: {caveat}" + "\n" for caveat in caveats]
    return out


def metric_order(metric):
    """METRIC_INFO order first, cold-start variants last, rest alphabetical."""
    known = list(METRIC_INFO)
    return (
        metric.startswith("cold-start-"),
        known.index(metric) if metric in known else len(known),
        metric,
    )


def render_details(results):
    """Every metric of every compared feature, behind a fold."""
    body = []
    for platform in results.platforms():
        features, sources = results.features(platform)
        for feature in sorted(features, key=str.upper):
            native = features[feature].get("NATIVE", {})
            wasm = features[feature].get("WASM", {})
            if not (native and wasm):
                continue
            rows = []
            for key in sorted(set(native) | set(wasm), key=metric_order):
                label, unit, higher = describe(key)
                n, w = native.get(key), wasm.get(key)
                if not n and not w:
                    # reportMetrics() reports a metric a test never produced
                    # as 0 rather than omitting it.
                    continue
                rows.append([
                    f"{label} ({unit})" if unit else label,
                    fmt(n),
                    fmt_wasm(w, gain(n, w, higher)),
                ])
            if rows:
                where = ", ".join(sorted(sources[feature]))
                body += [
                    f"#### {flat_name(feature)} -- {where} ({platform_name(platform)})",
                    "",
                ]
                body += table(
                    ["Metric", "native", "wasm"], rows, ["---", "---:", "---:"]
                )

    if not body:
        return []
    return (
        ["<details>", "<summary>All metrics, per compared feature</summary>", ""]
        + body
        + ["</details>", ""]
    )


def render_notes(results):
    """Series the comparison is not about, named once so nothing is silent."""
    lines = []

    skipped = defaultdict(set)
    untagged = defaultdict(set)
    for job in results.jobs.values():
        for feature, by_tag in job.metrics.items():
            for tag in by_tag:
                if tag in OTHER_TAGS:
                    skipped[OTHER_TAGS[tag]].add(feature)
        for name in job.untagged:
            untagged[name].add(job.family)

    lines += [
        f"- {len(features)} `{backend}` features have no native-vs-wasm "
        "comparison to make; their series are tracked in perfherder as usual."
        for backend, features in sorted(skipped.items())
    ]
    lines += [
        f"- `{name}` carries no backend tag, so it is not compared (from "
        + ", ".join(f"`{family}`" for family in sorted(families))
        + ")."
        for name, families in sorted(untagged.items())
    ]

    if not results.include_harness:
        present = {
            job.family
            for job in results.jobs.values()
            if job.family in HARNESS_FAMILIES
        }
        lines += [
            f"- Left out `{family}` series ({HARNESS_FAMILIES[family]}). Pass "
            "--include-harness-tests to put them back."
            for family in sorted(present)
        ]

    if not lines:
        return []
    return ["## Notes", ""] + lines + [""]


def render(results, source=None):
    out = ["# ML perftest report: native ONNX vs WASM", ""]
    if source:
        out += [f"Source: {source}", ""]

    out += render_coverage(results)
    out += render_summary(results)
    out += render_comparison(results)
    out += render_details(results)
    out += render_notes(results)

    return "\n".join(out).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--revision", help="Try push revision to pull logs from")
    # Not --log: `mach python` owns that prefix (--log-file, --log-interval)
    # and rejects the invocation as ambiguous before the script ever runs.
    source.add_argument(
        "--logs", nargs="+", type=Path, help="Local perftest log file(s)"
    )
    parser.add_argument("--repo", default="try", help="Treeherder repo (default: try)")
    parser.add_argument(
        "--filter",
        dest="filter_regex",
        default="-ml-",
        help="Job name filter regex (default: -ml-, which is what the perftest "
        "task names use and, unlike 'ml', does not also match mozlint)",
    )
    parser.add_argument(
        "--include-harness-tests",
        action="store_true",
        help="Also compare the engine template and multi-engine stress tests, "
        "which measure the ML plumbing rather than a feature",
    )
    parser.add_argument(
        "--output", type=Path, help="Write markdown here instead of stdout"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Where to keep downloaded logs. Reused on the next run for the "
        "same push; defaults to a fresh temporary directory.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refetch from Treeherder even if --cache-dir holds a download",
    )
    args = parser.parse_args()

    results = Results(include_harness=args.include_harness_tests)

    if args.revision:
        cache_dir = args.cache_dir or Path(tempfile.mkdtemp(prefix="ml-perf-logs-"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        data = fetch_from_treeherder(
            args.revision, args.repo, args.filter_regex, cache_dir, args.refresh
        )
        ingest_treeherder(results, data)
        print(f"Logs in {cache_dir}", file=sys.stderr)
        source = f"try push [{args.revision[:12]}](https://treeherder.mozilla.org/jobs?repo={args.repo}&revision={args.revision})"
    else:
        for path in args.logs:
            text = path.read_text(errors="replace")
            try:
                results.ingest_perfherder(path.name, json.loads(text))
            except json.JSONDecodeError:
                results.ingest_log(path.name, text)
        source = ", ".join(f"`{path}`" for path in args.logs)

    report = render(results, source)

    if args.output:
        args.output.write_text(report)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(report)


if __name__ == "__main__":
    main()
