#ifndef _MPS_UTILS_H_
#define _MPS_UTILS_H_
#include <torch/extension.h>
#ifdef __OBJC__
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

typedef id<MTLBuffer> MTLBuffer_t;
typedef id<MTLComputeCommandEncoder> MTLComputeCommandEncoder_t;
#else
typedef void* MTLBuffer;
typedef void* MTLBuffer_t;
typedef void* MTLComputeCommandEncoder;
typedef void* MTLComputeCommandEncoder_t;
#endif

// utils
static inline MTLBuffer_t getMTLBufferStorage(const at::Tensor& tensor) {
  return __builtin_bit_cast(MTLBuffer_t, tensor.storage().data());
}

template <typename T,
          std::enable_if_t<!std::is_same<std::decay_t<T>, at::Tensor>::value, bool> = true>
void setMTLArg(MTLComputeCommandEncoder_t encoder, int index, T&& t);

template <typename T,
          std::enable_if_t<std::is_same<std::decay_t<T>, at::Tensor>::value, bool> = true>
void setMTLArg(MTLComputeCommandEncoder_t encoder, int index, T&& t) {
  [encoder setBuffer:getMTLBufferStorage(t) offset:0 atIndex:index];
}

template <typename T, std::enable_if_t<!std::is_same<std::decay_t<T>, at::Tensor>::value, bool>>
void setMTLArg(MTLComputeCommandEncoder_t encoder, int index, T&& t) {
  [encoder setBytes:&t length:sizeof(t) atIndex:index];
}

inline void setMTLArgsImpl(MTLComputeCommandEncoder_t, int) {}

template <typename T, typename... Args>
void setMTLArgsImpl(MTLComputeCommandEncoder_t encoder, int index, T&& t, Args&&... args) {
  setMTLArg(encoder, index, std::forward<T>(t));
  setMTLArgsImpl(encoder, index + 1, std::forward<Args>(args)...);
}

template <typename... Args>
void setMTLArgs(MTLComputeCommandEncoder_t encoder, MTLComputePipelineState_t pso, Args&&... args) {
  [encoder setComputePipelineState:pso];
  setMTLArgsImpl(encoder, 0, std::forward<Args>(args)...);
}
#endif
