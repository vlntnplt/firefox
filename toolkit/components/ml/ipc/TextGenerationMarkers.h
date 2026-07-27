/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef mozilla_hwinference_TextGenerationMarkers_h
#define mozilla_hwinference_TextGenerationMarkers_h

#include "mozilla/ProfilerMarkers.h"

// Markers for the text-generation pipeline, two levels deep: the chrome
// wrapper marks whole create()/generate() spans on the parent main
// thread, and the HWInference child marks the load/prefill/decode
// phases on the generator's TaskQueue thread. The child phases reuse
// the timestamps kept for TextGenerationTimings, so the markers and the
// usage numbers returned to callers always agree.

// PROFILER_MARKER resolves marker type structs in this namespace.
namespace geckoprofiler::markers {

using namespace mozilla;

struct TextGenerationLoadMarker {
  static constexpr Span<const char> MarkerTypeName() {
    return MakeStringSpan("TextGeneratorLoad");
  }
  static void StreamJSONMarkerData(baseprofiler::SpliceableJSONWriter& aWriter,
                                   uint32_t aContextSize, uint32_t aThreads,
                                   const ProfilerString8View& aKvCacheDtype,
                                   const ProfilerString8View& aOutcome) {
    aWriter.IntProperty("contextSize", aContextSize);
    aWriter.IntProperty("threads", aThreads);
    aWriter.StringProperty("kvCacheDtype", aKvCacheDtype);
    aWriter.StringProperty("outcome", aOutcome);
  }
  static MarkerSchema MarkerTypeDisplay() {
    using MS = MarkerSchema;
    MS schema{MS::Location::MarkerChart, MS::Location::MarkerTable};
    schema.SetChartLabel("ctx {marker.data.contextSize}");
    schema.SetTableLabel(
        "ctx {marker.data.contextSize}, {marker.data.threads} threads, kv "
        "{marker.data.kvCacheDtype}: {marker.data.outcome}");
    schema.AddKeyLabelFormat("contextSize", "Context size",
                             MS::Format::Integer);
    schema.AddKeyLabelFormat("threads", "Threads", MS::Format::Integer);
    schema.AddKeyLabelFormat("kvCacheDtype", "KV cache dtype",
                             MS::Format::String);
    schema.AddKeyLabelFormat("outcome", "Outcome", MS::Format::String);
    return schema;
  }
};

struct TextGenerationPrefillMarker {
  static constexpr Span<const char> MarkerTypeName() {
    return MakeStringSpan("TextGeneratorPrefill");
  }
  static void StreamJSONMarkerData(baseprofiler::SpliceableJSONWriter& aWriter,
                                   uint32_t aPromptTokens) {
    aWriter.IntProperty("promptTokens", aPromptTokens);
  }
  static MarkerSchema MarkerTypeDisplay() {
    using MS = MarkerSchema;
    MS schema{MS::Location::MarkerChart, MS::Location::MarkerTable};
    schema.SetChartLabel("{marker.data.promptTokens} prompt tokens");
    schema.SetTableLabel("{marker.data.promptTokens} prompt tokens");
    schema.AddKeyLabelFormat("promptTokens", "Prompt tokens",
                             MS::Format::Integer);
    return schema;
  }
};

struct TextGenerationDecodeMarker {
  static constexpr Span<const char> MarkerTypeName() {
    return MakeStringSpan("TextGeneratorDecode");
  }
  static void StreamJSONMarkerData(baseprofiler::SpliceableJSONWriter& aWriter,
                                   uint32_t aGeneratedTokens,
                                   const ProfilerString8View& aReason) {
    aWriter.IntProperty("generatedTokens", aGeneratedTokens);
    aWriter.StringProperty("reason", aReason);
  }
  static MarkerSchema MarkerTypeDisplay() {
    using MS = MarkerSchema;
    MS schema{MS::Location::MarkerChart, MS::Location::MarkerTable};
    schema.SetChartLabel("{marker.data.generatedTokens} tokens");
    schema.SetTableLabel(
        "{marker.data.generatedTokens} tokens ({marker.data.reason})");
    schema.AddKeyLabelFormat("generatedTokens", "Generated tokens",
                             MS::Format::Integer);
    schema.AddKeyLabelFormat("reason", "Finish reason", MS::Format::String);
    return schema;
  }
};

struct TextGenerationCreateMarker {
  static constexpr Span<const char> MarkerTypeName() {
    return MakeStringSpan("TextGeneratorCreate");
  }
  static void StreamJSONMarkerData(baseprofiler::SpliceableJSONWriter& aWriter,
                                   const ProfilerString8View& aOutcome) {
    aWriter.StringProperty("outcome", aOutcome);
  }
  static MarkerSchema MarkerTypeDisplay() {
    using MS = MarkerSchema;
    MS schema{MS::Location::MarkerChart, MS::Location::MarkerTable};
    schema.SetTableLabel("{marker.data.outcome}");
    schema.AddKeyLabelFormat("outcome", "Outcome", MS::Format::String);
    return schema;
  }
};

struct TextGenerationGenerateMarker {
  static constexpr Span<const char> MarkerTypeName() {
    return MakeStringSpan("TextGeneratorGenerate");
  }
  static void StreamJSONMarkerData(baseprofiler::SpliceableJSONWriter& aWriter,
                                   uint32_t aPromptTokens,
                                   uint32_t aGeneratedTokens,
                                   const ProfilerString8View& aReason) {
    aWriter.IntProperty("promptTokens", aPromptTokens);
    aWriter.IntProperty("generatedTokens", aGeneratedTokens);
    aWriter.StringProperty("reason", aReason);
  }
  static MarkerSchema MarkerTypeDisplay() {
    using MS = MarkerSchema;
    MS schema{MS::Location::MarkerChart, MS::Location::MarkerTable};
    schema.SetChartLabel("{marker.data.generatedTokens} tokens");
    schema.SetTableLabel(
        "{marker.data.promptTokens} prompt + {marker.data.generatedTokens} "
        "generated tokens ({marker.data.reason})");
    schema.AddKeyLabelFormat("promptTokens", "Prompt tokens",
                             MS::Format::Integer);
    schema.AddKeyLabelFormat("generatedTokens", "Generated tokens",
                             MS::Format::Integer);
    schema.AddKeyLabelFormat("reason", "Finish reason", MS::Format::String);
    return schema;
  }
};

}  // namespace geckoprofiler::markers

#endif  // mozilla_hwinference_TextGenerationMarkers_h
