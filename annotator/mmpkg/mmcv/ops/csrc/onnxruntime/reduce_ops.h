// Copyright (c) OpenMMLab. All rights reserved
#ifndef ONNXRUNTIME_REDUCE_OPS_H
#define ONNXRUNTIME_REDUCE_OPS_H

#include <onnxruntime_cxx_api.h>

struct MMCVCumMaxKernel {
 public:
  MMCVCumMaxKernel(Ort::CustomOpApi ort, const OrtKernelInfo* info)
      : ort_(ort) {
    dim_ = ort_.KernelInfoGetAttribute<int64_t>(info, "dim");

    // create allocator
    allocator_ = Ort::AllocatorWithDefaultOptions();
  }

  void Compute(OrtKernelContext* context);

 private:
  Ort::CustomOpApi ort_;
  Ort::AllocatorWithDefaultOptions allocator_;

  int64_t dim_;
};

struct MMCVCumMinKernel {
 public:
  MMCVCumMinKernel(Ort::CustomOpApi ort, const OrtKernelInfo* info)
      : ort_(ort) {
    dim_ = ort_.KernelInfoGetAttribute<int64_t>(info, "dim");

    // create allocator
    allocator_ = Ort::AllocatorWithDefaultOptions();
  }

  void Compute(OrtKernelContext* context);

 private:
  Ort::CustomOpApi ort_;
  Ort::AllocatorWithDefaultOptions allocator_;

  int64_t dim_;
};

struct MMCVCumMaxCustomOp
    : Ort::CustomOpBase<MMCVCumMaxCustomOp, MMCVCumMaxKernel> {
  void* CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo* info) const {
    return new MMCVCumMaxKernel(api, info);
  }

  const char* GetName() const { return "cummax"; }

  size_t GetInputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetInputType(size_t) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

  size_t GetOutputTypeCount() const { return 2; }
  ONNXTensorElementDataType GetOutputType(size_t index) const {
    if (index == 1) return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

  // force cpu
  const char* GetExecutionProviderType() const {
    return "CPUExecutionProvider";
  };
};

struct MMCVCumMinCustomOp
    : Ort::CustomOpBase<MMCVCumMinCustomOp, MMCVCumMinKernel> {
  void* CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo* info) const {
    return new MMCVCumMinKernel(api, info);
  }

  const char* GetName() const { return "cummin"; }

  size_t GetInputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetInputType(size_t) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

  size_t GetOutputTypeCount() const { return 2; }
  ONNXTensorElementDataType GetOutputType(size_t index) const {
    if (index == 1) return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

  // force cpu
  const char* GetExecutionProviderType() const {
    return "CPUExecutionProvider";
  };
};

#endif  // ONNXRUNTIME_REDUCE_OPS_H
