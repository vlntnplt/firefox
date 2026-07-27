/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "TextGenerationChild.h"

#include <cstdio>
#include <memory>

#include "HWInferenceLog.h"
#include "LlamaBackend.h"
#include "TextGenerationMarkers.h"
#include "mozilla/CheckedInt.h"
#include "mozilla/Encoding.h"
#include "mozilla/Logging.h"
#include "mozilla/ScopeExit.h"
#include "mozilla/SharedThreadPool.h"
#include "mozilla/Span.h"
#include "mozilla/TimeStamp.h"
#include "mozilla/UniquePtrExtensions.h"
#include "mozilla/dom/BindingUtils.h"
#include "nsThreadUtils.h"

#ifdef XP_WIN
#  include <fcntl.h>
#  include <io.h>
#endif

namespace mozilla::hwinference {

#define LOGD(fmt, ...) \
  MOZ_LOG_FMT(gHWInferenceLog, LogLevel::Debug, fmt, ##__VA_ARGS__)

using dom::LlamaChatMessage;
using dom::LlamaChatOptions;
using dom::LlamaChatPhase;
using dom::LlamaChatResponse;
using dom::LlamaFormatChatOptions;
using dom::LlamaKVCacheDtype;
using dom::LlamaModelOptions;
using dom::LlamaSamplerConfig;
using dom::LlamaSamplerType;
using dom::TextGenerationFinishReason;

// Bridges to the LlamaRunner-era backend vocabulary; both retire when
// the backend dictionaries unify onto the surface enums.
static LlamaSamplerType ToLlamaSamplerType(dom::TextGenerationSamplerType aType) {
  switch (aType) {
    case dom::TextGenerationSamplerType::Logit_bias:
      return LlamaSamplerType::Logit_bias;
    case dom::TextGenerationSamplerType::Top_k:
      return LlamaSamplerType::Top_k;
    case dom::TextGenerationSamplerType::Top_p:
      return LlamaSamplerType::Top_p;
    case dom::TextGenerationSamplerType::Min_p:
      return LlamaSamplerType::Min_p;
    case dom::TextGenerationSamplerType::Temperature:
      return LlamaSamplerType::Temperature;
    case dom::TextGenerationSamplerType::Dist:
      return LlamaSamplerType::Dist;
  }
  MOZ_CRASH("bad TextGenerationSamplerType");
}

static LlamaKVCacheDtype ToLlamaKVCacheDtype(
    dom::TextGenerationKVCacheDtype aDtype) {
  switch (aDtype) {
    case dom::TextGenerationKVCacheDtype::F32:
      return LlamaKVCacheDtype::F32;
    case dom::TextGenerationKVCacheDtype::F16:
      return LlamaKVCacheDtype::F16;
    case dom::TextGenerationKVCacheDtype::Q8_0:
      return LlamaKVCacheDtype::Q8_0;
    case dom::TextGenerationKVCacheDtype::Q5_1:
      return LlamaKVCacheDtype::Q5_1;
    case dom::TextGenerationKVCacheDtype::Q5_0:
      return LlamaKVCacheDtype::Q5_0;
    case dom::TextGenerationKVCacheDtype::Q4_1:
      return LlamaKVCacheDtype::Q4_1;
    case dom::TextGenerationKVCacheDtype::Q4_0:
      return LlamaKVCacheDtype::Q4_0;
  }
  MOZ_CRASH("bad TextGenerationKVCacheDtype");
}

static LlamaFormatChatOptions ToFormatChatOptions(
    const CopyableTArray<ChatMessage>& aHistory) {
  LlamaFormatChatOptions options;
  for (const ChatMessage& message : aHistory) {
    LlamaChatMessage* entry = options.mMessages.AppendElement(fallible);
    MOZ_RELEASE_ASSERT(entry);
    entry->mRole = dom::GetEnumString(message.role());
    entry->mContent = message.content();
  }
  return options;
}

static LlamaChatOptions ToLlamaChatOptions(const GenerateRequest& aRequest,
                                           nsCString&& aFormattedPrompt) {
  LlamaChatOptions options;
  options.mPrompt = std::move(aFormattedPrompt);
  options.mMaxGeneratedTokens = static_cast<int32_t>(aRequest.maxTokens());
  options.mStopOnEndOfGenerationTokens = aRequest.stopOnEndOfGenerationTokens();
  // The Generation buffers deltas itself; the backend reports every
  // token.
  options.mMinOutputBufferSize = 1;
  for (int32_t stopToken : aRequest.stopTokens()) {
    MOZ_RELEASE_ASSERT(options.mStopTokens.AppendElement(stopToken, fallible));
  }
  for (const Sampler& sampler : aRequest.samplers()) {
    LlamaSamplerConfig* config = options.mSamplers.AppendElement(fallible);
    MOZ_RELEASE_ASSERT(config);
    config->mType = ToLlamaSamplerType(sampler.type());
    config->mTopK = sampler.topK();
    config->mTopP = sampler.topP();
    config->mTemp = sampler.temp();
    config->mSeed.Construct(sampler.seed());
    for (const LogitBias& bias : sampler.logitBias()) {
      dom::LlamaLogitBias* entry = config->mLogitBias.AppendElement(fallible);
      MOZ_RELEASE_ASSERT(entry);
      entry->mToken = bias.token();
      entry->mBias = bias.bias();
    }
  }
  return options;
}

// Decode `aBytes` through the streaming UTF-8 `aDecoder`, appending the
// valid UTF-8 output to `aOut`. An incomplete trailing sequence is
// retained inside the decoder until a later call completes it; bytes
// that can never be valid are replaced with U+FFFD. llama.cpp
// byte-fallback tokens split codepoints across pieces, and this is what
// keeps every delta and the reply valid UTF-8 (regression contract of
// bug 2043430).
[[nodiscard]] static bool AppendDecodedUtf8(Decoder& aDecoder,
                                            const nsACString& aBytes,
                                            nsACString& aOut) {
  Span<const uint8_t> src(
      reinterpret_cast<const uint8_t*>(aBytes.BeginReading()), aBytes.Length());
  CheckedInt<size_t> capacity = aDecoder.MaxUTF8BufferLength(src.Length());
  size_t base = aOut.Length();
  if (!capacity.isValid() ||
      !aOut.SetLength(base + capacity.value(), fallible)) {
    return false;
  }
  Span<uint8_t> dst(reinterpret_cast<uint8_t*>(aOut.BeginWriting()) + base,
                    capacity.value());
  aOut.SetLength(base + std::get<2>(aDecoder.DecodeToUTF8(src, dst, false)));
  return true;
}

// One Generate request: the per-request state and the compute that runs
// on the child's TaskQueue. Events and the reply hop to the actor thread
// through the child, reply last, preserving the protocol's ordering
// contract.
class TextGenerationChild::Generation {
 public:
  NS_INLINE_DECL_THREADSAFE_REFCOUNTING(Generation);

  Generation(TextGenerationChild* aChild,
             CopyableTArray<ChatMessage>&& aHistory, GenerateRequest&& aRequest,
             GenerateResolver&& aResolve)
      : mChild(aChild),
        mHistory(std::move(aHistory)),
        mRequest(std::move(aRequest)),
        mResolver(std::make_shared<GenerateResolver>(std::move(aResolve))),
        mBufferLength(std::max(1u, mRequest.bufferLength())),
        mDecoder(UTF_8_ENCODING->NewDecoderWithoutBOMHandling()) {}

  void Run() {
    MOZ_ASSERT(mChild->mTaskQueue->IsOnCurrentThread());
    if (!mChild->mBackend) {
      Reply(GenerateResponse(GenerateError(mChild->mLoadError)));
      return;
    }

    auto formatted =
        mChild->mBackend->FormatChat(ToFormatChatOptions(mHistory));
    if (formatted.isErr()) {
      Reply(GenerateResponse(GenerateError(formatted.unwrapErr().mMessage)));
      return;
    }

    mGenerateStart = TimeStamp::Now();
    auto status = mChild->mBackend->Generate(
        ToLlamaChatOptions(mRequest, formatted.unwrap()),
        [self = RefPtr{this}](const LlamaChatResponse& aChunk) {
          return self->HandleChunk(aChunk);
        },
        [self = RefPtr{this}]() -> bool {
          return self->mCancelled || self->mChild->mShutdown;
        });
    Flush();
    if (status.isErr()) {
      Reply(GenerateResponse(GenerateError(status.unwrapErr().mMessage)));
      return;
    }

    TimeStamp end = TimeStamp::Now();
    if (mPrefillEnd.IsNull()) {
      mPrefillEnd = end;
    }
    TextGenerationFinishReason reason = ComputeFinishReason();
    PROFILER_MARKER("TextGenerator prefill", ML_INFERENCE,
                    MarkerTiming::Interval(mGenerateStart, mPrefillEnd),
                    TextGenerationPrefillMarker, mPromptTokens);
    PROFILER_MARKER("TextGenerator decode", ML_INFERENCE,
                    MarkerTiming::Interval(mPrefillEnd, end),
                    TextGenerationDecodeMarker, mGeneratedTokens,
                    dom::GetEnumString(reason));
    Timings timings((mPrefillEnd - mGenerateStart).ToMilliseconds(),
                    (end - mPrefillEnd).ToMilliseconds());
    Reply(GenerateResponse(GenerateResult(
        mContent, reason, Usage(mPromptTokens, mGeneratedTokens, timings))));
  }

  // Any thread. The compute loop polls the flag between chunks.
  void RequestCancel() { mCancelled = true; }

 private:
  ~Generation() = default;

  llama::ResultStatus HandleChunk(const LlamaChatResponse& aChunk) {
    if (aChunk.mPhase == LlamaChatPhase::Prompt) {
      mPromptTokens += aChunk.mTokens.Length();
      if (aChunk.mIsPhaseCompleted) {
        mPrefillEnd = TimeStamp::Now();
      }
      return Ok();
    }
    mGeneratedTokens += aChunk.mTokens.Length();
    if (!aChunk.mTokens.IsEmpty()) {
      mLastToken = aChunk.mTokens.LastElement();
    }
    if (!AppendDecodedUtf8(*mDecoder, aChunk.mPiece, mPendingDelta)) {
      return Err(
          llama::Error{"TextGenerator: failed to buffer decoded output"_ns});
    }
    mPendingTokens += aChunk.mTokens.Length();
    if (mPendingTokens >= mBufferLength) {
      Flush();
    }
    return Ok();
  }

  // Sends the buffered delta, which only ever holds complete valid UTF-8:
  // an unfinished trailing sequence waits inside the decoder and is
  // dropped if the generation ends first. The reply content is the
  // flushed deltas joined, so streamed text equals reply text exactly.
  void Flush() {
    mPendingTokens = 0;
    if (mPendingDelta.IsEmpty()) {
      return;
    }
    nsCString delta = std::move(mPendingDelta);
    mPendingDelta.Truncate();
    mContent.Append(delta);
    mChild->RunOnActorThread("TextGenerationChild::Delta",
                             [child = mChild, delta = std::move(delta)] {
                               (void)child->SendDelta(delta);
                             });
  }

  TextGenerationFinishReason ComputeFinishReason() const {
    if (mCancelled || mChild->mShutdown) {
      return TextGenerationFinishReason::Cancelled;
    }
    if (mGeneratedTokens >= mRequest.maxTokens()) {
      return TextGenerationFinishReason::Length;
    }
    if (mLastToken >= 0 && mRequest.stopTokens().Contains(mLastToken)) {
      return TextGenerationFinishReason::Stop_token;
    }
    return TextGenerationFinishReason::Eos;
  }

  void Reply(GenerateResponse&& aResponse) {
    mChild->RunOnActorThread(
        "TextGenerationChild::GenerateReply",
        [resolver = mResolver, response = std::move(aResponse)]() mutable {
          (*resolver)(std::move(response));
        });
  }

  const RefPtr<TextGenerationChild> mChild;
  const CopyableTArray<ChatMessage> mHistory;
  const GenerateRequest mRequest;
  const std::shared_ptr<GenerateResolver> mResolver;
  const uint32_t mBufferLength;
  const UniquePtr<Decoder> mDecoder;

  Atomic<bool> mCancelled{false};

  nsCString mContent;
  nsCString mPendingDelta;
  uint32_t mPendingTokens = 0;
  uint32_t mPromptTokens = 0;
  uint32_t mGeneratedTokens = 0;
  int32_t mLastToken = -1;
  TimeStamp mGenerateStart;
  TimeStamp mPrefillEnd;
};

TextGenerationChild::TextGenerationChild(const ipc::FileDescriptor& aModel,
                                         const TextGenerationOptions& aOptions)
    : mModel(aModel),
      mOptions(aOptions),
      mTaskQueue(TaskQueue::Create(SharedThreadPool::Get("TextGenerator", 1),
                                   "TextGenerationChild::mTaskQueue")),
      mActorThread(GetCurrentSerialEventTarget()) {}

TextGenerationChild::~TextGenerationChild() = default;

void TextGenerationChild::Initialize() {
  RefPtr<TextGenerationChild> self = this;
  MOZ_ALWAYS_SUCCEEDS(mTaskQueue->Dispatch(
      NS_NewRunnableFunction("TextGenerationChild::Load", [self] {
        TimeStamp loadStart = TimeStamp::Now();
        LoadResult result = self->LoadOnQueue();
        PROFILER_MARKER("TextGenerator load", ML_SETUP,
                        MarkerTiming::IntervalUntilNowFrom(loadStart),
                        TextGenerationLoadMarker, self->mOptions.contextSize(),
                        self->mOptions.numThreads(),
                        dom::GetEnumString(self->mOptions.kvCacheDtype()),
                        result.type() == LoadResult::TLoadSuccess
                            ? "ok"_ns
                            : self->mLoadError);
        self->RunOnActorThread("TextGenerationChild::Ready",
                               [self, result = std::move(result)] {
                                 (void)self->SendReady(result);
                               });
      })));
}

LoadResult TextGenerationChild::LoadOnQueue() {
  MOZ_ASSERT(mTaskQueue->IsOnCurrentThread());
  MOZ_ASSERT(!mBackend);
  mLoadError.AssignLiteral("TextGenerator: model load failed");

  TimeStamp loadStart = TimeStamp::Now();

  if (!mModel.IsValid()) {
    mLoadError.AssignLiteral("TextGenerator: invalid model file descriptor");
    return LoadResult(LoadError(mLoadError));
  }

  // fdopen()/_open_osfhandle() take ownership of the handle they are
  // given and fclose() closes it; hand them a duplicate so the
  // FileDescriptor's own handle is not double-closed.
  UniqueFileHandle dupHandle =
      DuplicateFileHandle(mModel.ClonePlatformHandle().get());
  if (!dupHandle) {
    mLoadError.AssignLiteral("TextGenerator: DuplicateFileHandle failed");
    return LoadResult(LoadError(mLoadError));
  }

#ifdef XP_WIN
  int fd = _open_osfhandle(dupHandle.release(), _O_RDONLY);
  if (fd == -1) {
    mLoadError.AssignLiteral("TextGenerator: _open_osfhandle failed");
    return LoadResult(LoadError(mLoadError));
  }
  FILE* fp = fdopen(fd, "rb");
  if (!fp) {
    _close(fd);
  }
#else
  FILE* fp = fdopen(dupHandle.release(), "r");
#endif
  if (!fp) {
    mLoadError.AssignLiteral("TextGenerator: fdopen failed");
    return LoadResult(LoadError(mLoadError));
  }
  auto closeFp = MakeScopeExit([fp] { fclose(fp); });

  LlamaModelOptions modelOptions;
  // mmap of a descriptor needs seccomp calls the utility sandbox denies,
  // and no consumer enables it; pinned off, with no knob in the surface.
  modelOptions.mUseMmap = false;
  modelOptions.mUseMlock = false;
  modelOptions.mContext.mNCtx = mOptions.contextSize();
  modelOptions.mContext.mNBatch = mOptions.batchSize();
  modelOptions.mContext.mNUbatch = mOptions.ubatchSize();
  // Copied verbatim; the wire always carries concrete counts (the
  // WebIDL layer resolves the 0 defaults before construction).
  modelOptions.mContext.mNThreads =
      static_cast<int32_t>(mOptions.numThreadsDecoding());
  modelOptions.mContext.mNThreadsBatch =
      static_cast<int32_t>(mOptions.numThreads());
  modelOptions.mContext.mKCacheDtype = ToLlamaKVCacheDtype(mOptions.kvCacheDtype());
  modelOptions.mContext.mVCacheDtype = ToLlamaKVCacheDtype(mOptions.kvCacheDtype());
  modelOptions.mContext.mFlashAttn = mOptions.flashAttn();

  RefPtr<llama::LlamaBackend> backend = new llama::LlamaBackend();
  auto result = backend->Reinitialize(modelOptions, fp);
  if (result.isErr()) {
    if (!result.unwrapErr().mMessage.IsEmpty()) {
      mLoadError = result.unwrapErr().mMessage;
    }
    return LoadResult(LoadError(mLoadError));
  }

  mBackend = std::move(backend);
  mLoadError.Truncate();
  double loadMs = (TimeStamp::Now() - loadStart).ToMilliseconds();
  LOGD("[{} - {}] model loaded in {}ms", fmt::ptr(this), __func__, loadMs);
  return LoadResult(LoadSuccess(loadMs));
}

void TextGenerationChild::RunOnActorThread(const char* aName,
                                           std::function<void()>&& aFn) {
  RefPtr<TextGenerationChild> self = this;
  MOZ_ALWAYS_SUCCEEDS(mActorThread->Dispatch(
      NS_NewRunnableFunction(aName, [self, fn = std::move(aFn)] {
        if (self->CanSend()) {
          fn();
        }
      })));
}

ipc::IPCResult TextGenerationChild::RecvGenerate(
    const GenerateRequest& aRequest, GenerateResolver&& aResolve) {
  LOGD("[{} - {}] {} message(s)", fmt::ptr(this), __func__,
       aRequest.messages().Length());

  mHistory.AppendElements(aRequest.messages());

  RefPtr<Generation> generation =
      new Generation(this, CopyableTArray<ChatMessage>(mHistory),
                     GenerateRequest(aRequest), std::move(aResolve));
  mCurrentGeneration = generation;
  MOZ_ALWAYS_SUCCEEDS(mTaskQueue->Dispatch(NS_NewRunnableFunction(
      "TextGenerationChild::Generate", [generation] { generation->Run(); })));
  return IPC_OK();
}

ipc::IPCResult TextGenerationChild::RecvClear() {
  LOGD("[{} - {}]", fmt::ptr(this), __func__);
  mHistory.Clear();
  return IPC_OK();
}

ipc::IPCResult TextGenerationChild::RecvCancel() {
  LOGD("[{} - {}]", fmt::ptr(this), __func__);
  if (mCurrentGeneration) {
    mCurrentGeneration->RequestCancel();
  }
  return IPC_OK();
}

void TextGenerationChild::ActorDestroy(ActorDestroyReason aReason) {
  LOGD("[{} - {}]", fmt::ptr(this), __func__);
  mShutdown = true;
  mCurrentGeneration = nullptr;
  mTaskQueue->BeginShutdown();
}

#undef LOGD
#undef LOGE

}  // namespace mozilla::hwinference
