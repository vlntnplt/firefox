/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/** JS-side counterpart of MLProfilerMarkers.h. */

export const AIR_TEXT_GENERATION_TRACK = "AIR:TextGeneration";

const MARKER_SCHEMAS = [
  {
    name: "MLEngineCreate",
    display: ["marker-chart", "marker-table", "timeline-overview"],
    chartLabel: "engine create",
    tableLabel: "engine create",
    tooltipLabel: "engine create",
    data: [
      { key: "fetchMs", label: "Model fetch", format: "duration" },
      { key: "generatorMs", label: "Generator create", format: "duration" },
      { key: "overheadMs", label: "Overhead", format: "duration" },
      {
        key: "featureId",
        label: "Feature",
        format: "string",
        searchable: true,
      },
      { key: "engineId", label: "Engine", format: "string", searchable: true },
      { key: "modelId", label: "Model", format: "string", searchable: true },
    ],
  },
  {
    name: "MLModelFetch",
    display: ["marker-chart", "marker-table"],
    chartLabel: "model fetch: {marker.data.bytes}",
    tableLabel: "model fetch: {marker.data.bytes}",
    tooltipLabel: "model fetch",
    data: [
      { key: "bytes", label: "Model size", format: "bytes" },
      { key: "modelId", label: "Model", format: "string", searchable: true },
    ],
  },
  {
    name: "MLFailed",
    display: ["marker-chart", "marker-table", "timeline-overview"],
    chartLabel: "failed: {marker.data.phase}",
    tableLabel: "failed: {marker.data.phase}",
    tooltipLabel: "failed",
    data: [
      { key: "phase", label: "Phase", format: "string" },
      { key: "reason", label: "Reason", format: "string" },
    ],
  },
  {
    name: "MLEngineRun",
    display: ["marker-chart", "marker-table", "timeline-overview"],
    chartLabel: "engine run: {marker.data.outputTokens} tokens",
    tableLabel: "engine run: {marker.data.outputTokens} tokens",
    tooltipLabel: "engine run",
    data: [
      { key: "streaming", label: "Streamed", format: "integer" },
      { key: "inputTokens", label: "Prompt tokens", format: "integer" },
      { key: "outputTokens", label: "Generated tokens", format: "integer" },
      { key: "ttfcMs", label: "Time to first chunk", format: "duration" },
      { key: "chunkTokens", label: "Tokens per chunk", format: "integer" },
      { key: "computeMs", label: "Model compute", format: "duration" },
      { key: "overheadMs", label: "Overhead", format: "duration" },
      { key: "reason", label: "Finish reason", format: "string" },
      {
        key: "featureId",
        label: "Feature",
        format: "string",
        searchable: true,
      },
    ],
  },
];

let gSchemasRegistered = false;

export function registerMarkerSchemas() {
  if (gSchemasRegistered) {
    return;
  }
  gSchemasRegistered = true;
  for (const schema of MARKER_SCHEMAS) {
    ChromeUtils.registerMarkerSchema(schema);
  }
}

export function mark(startTime, payload) {
  ChromeUtils.addProfilerMarker(
    AIR_TEXT_GENERATION_TRACK,
    { startTime, category: "ML" },
    payload
  );
}

export function markFailed(startTime, phase, error) {
  mark(startTime, {
    type: "MLFailed",
    phase,
    reason: String(error?.name ?? error),
  });
}
