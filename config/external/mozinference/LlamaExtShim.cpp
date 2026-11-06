/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// llama-ext.h is llama.cpp's staging API: C++ linkage, so the symbols file
// cannot name it. These wrappers give the runtime linker a C ABI.

#include "llama-ext.h"
#include "mozilla/Types.h"

extern "C" {

MOZ_EXPORT void mozllama_set_embeddings_nextn(llama_context* ctx, bool value,
                                              bool masked) {
  llama_set_embeddings_nextn(ctx, value, masked);
}

MOZ_EXPORT float* mozllama_get_embeddings_nextn_ith(llama_context* ctx,
                                                    int32_t i) {
  return llama_get_embeddings_nextn_ith(ctx, i);
}

}  // extern "C"
