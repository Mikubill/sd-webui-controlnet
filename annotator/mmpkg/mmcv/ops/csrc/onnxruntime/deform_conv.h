// Copyright (c) OpenMMLab. All rights reserved
#ifndef ONNXRUNTIME_DEFORM_CONV_H
#define ONNXRUNTIME_DEFORM_CONV_H

#include <onnxruntime_cxx_api.h>

struct MMCVDeformConvKernel {
  MMCVDeformConvKernel(OrtApi api, const OrtKernelInfo *info);

  void Compute(OrtKernelContext *context);

 protected:
  OrtApi api_;
  Ort::CustomOpApi ort_;
  const OrtKernelInfo *info_;
  Ort::AllocatorWithDefaultOptions allocator_;

  int64_t stride_height_;
  int64_t stride_width_;
  int64_t padding_height_;
  int64_t padding_width_;
  int64_t dilation_height_;
  int64_t dilation_width_;
  int64_t deformable_group_;
  int64_t group_;
  int64_t im2col_step_;
};

struct MMCVDeformConvOp
    : Ort::CustomOpBase<MMCVDeformConvOp, MMCVDeformConvKernel> {
  void *CreateKernel(OrtApi api, const OrtKernelInfo *info) const {
    return new MMCVDeformConvKernel(api, info);
  }

  const char *GetName() const { return "MMCVDeformConv2d"; };

  size_t GetInputTypeCount() const { return 3; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(
      size_t index) const {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  }

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

  // force cpu
  const char *GetExecutionProviderType() const {
    return "CPUExecutionProvider";
  };
};
#endif
