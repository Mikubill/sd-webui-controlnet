// Copyright (c) OpenMMLab. All rights reserved
#ifndef ONNXRUNTIME_SOFT_NMS_H
#define ONNXRUNTIME_SOFT_NMS_H
#include <onnxruntime_cxx_api.h>

struct SoftNmsKernel {
  SoftNmsKernel(OrtApi api, const OrtKernelInfo *info);

  void Compute(OrtKernelContext *context);

 protected:
  OrtApi api_;
  Ort::CustomOpApi ort_;
  const OrtKernelInfo *info_;
  Ort::AllocatorWithDefaultOptions allocator_;

  float iou_threshold_;
  float sigma_;
  float min_score_;
  int64_t method_;
  int64_t offset_;
};

struct SoftNmsOp : Ort::CustomOpBase<SoftNmsOp, SoftNmsKernel> {
  void *CreateKernel(OrtApi api, const OrtKernelInfo *info) const {
    return new SoftNmsKernel(api, info);
  };

  const char *GetName() const { return "SoftNonMaxSuppression"; };

  size_t GetInputTypeCount() const { return 2; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

  size_t GetOutputTypeCount() const { return 2; };
  ONNXTensorElementDataType GetOutputType(size_t index) const {
    if (index == 1) {
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    }
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

  // force cpu
  const char *GetExecutionProviderType() const {
    return "CPUExecutionProvider";
  };
};
#endif  // ONNXRUNTIME_SOFT_NMS_H
