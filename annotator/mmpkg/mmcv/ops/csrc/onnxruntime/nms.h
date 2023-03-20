// Copyright (c) OpenMMLab. All rights reserved
#ifndef ONNXRUNTIME_NMS_H
#define ONNXRUNTIME_NMS_H

#include <onnxruntime_cxx_api.h>

struct NmsKernel {
  NmsKernel(OrtApi api, const OrtKernelInfo *info);

  void Compute(OrtKernelContext *context);

 protected:
  OrtApi api_;
  Ort::CustomOpApi ort_;
  const OrtKernelInfo *info_;
  Ort::AllocatorWithDefaultOptions allocator_;

  float iou_threshold_;
  int64_t offset_;
};

struct NmsOp : Ort::CustomOpBase<NmsOp, NmsKernel> {
  void *CreateKernel(OrtApi api, const OrtKernelInfo *info) const {
    return new NmsKernel(api, info);
  };

  const char *GetName() const { return "NonMaxSuppression"; };

  size_t GetInputTypeCount() const { return 2; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t index) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  }

  // force cpu
  const char *GetExecutionProviderType() const {
    return "CPUExecutionProvider";
  }
};

#endif
