/* -*- Mode: IDL; tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/**
 * The vocabulary of the text-generation wire (PTextGeneration): these
 * enums are carried on the wire directly and are the one enum family
 * from chrome JS to the backend. The chrome-only TextGenerator surface
 * over them follows in its own commit.
 */

enum TextGenerationRole {
  "system",
  "user",
  "assistant",
};

enum TextGenerationSamplerType {
  "logit-bias",
  "top-k",
  "top-p",
  "temperature",
  "dist",
};

enum TextGenerationFinishReason {
  "eos",
  "length",
  /**
   * Never reported today: the backend stops before delivering the token it
   * stopped on, so a run ended by a stop token reports "eos".
   */
  "stop-token",
  "cancelled",
};

enum TextGenerationKVCacheDtype {
  "f32",
  "f16",
  "q8_0",
  "q5_1",
  "q5_0",
  "q4_1",
  "q4_0",
};
