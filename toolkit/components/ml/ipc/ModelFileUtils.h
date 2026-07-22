/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef mozilla_hwinference_ModelFileUtils_h
#define mozilla_hwinference_ModelFileUtils_h

#include "js/RootingAPI.h"
#include "js/TypeDecls.h"
#include "nsError.h"

namespace mozilla::ipc {
class FileDescriptor;
}

namespace mozilla::hwinference {

// Extracts the OS-level descriptor backing a file-backed DOM Blob (such as
// the ones ModelHub materializes) into an ipc::FileDescriptor, which
// duplicates the handle and can be sent to the utility process. Fails if
// the Blob's stream is not file-backed.
nsresult BlobJSObjectToFileDescriptor(JSContext* aCx,
                                      JS::Handle<JS::Value> aValue,
                                      ipc::FileDescriptor* aDesc);

}  // namespace mozilla::hwinference

#endif  // mozilla_hwinference_ModelFileUtils_h
