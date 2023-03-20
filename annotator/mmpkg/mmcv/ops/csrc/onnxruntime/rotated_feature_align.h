#ifndef ONNXRUNTIME_ROTATED_FEATURE_ALIGN_H
#define ONNXRUNTIME_ROTATED_FEATURE_ALIGN_H

#include <onnxruntime_cxx_api.h>

#include <cmath>

struct MMCVRotatedFeatureAlignKernel {
 public:
  MMCVRotatedFeatureAlignKernel(Ort::CustomOpApi ort, const OrtKernelInfo* info)
      : ort_(ort) {
    spatial_scale_ = ort_.KernelInfoGetAttribute<float>(info, "spatial_scale");
    points_ = ort_.KernelInfoGetAttribute<int64_t>(info, "points");
  }

  void Compute(OrtKernelContext* context);

 private:
  Ort::CustomOpApi ort_;
  float spatial_scale_;
  int points_;
};

struct MMCVRotatedFeatureAlignCustomOp
    : Ort::CustomOpBase<MMCVRotatedFeatureAlignCustomOp,
                        MMCVRotatedFeatureAlignKernel> {
  void* CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo* info) const {
    return new MMCVRotatedFeatureAlignKernel(api, info);
  }

  const char* GetName() const { return "MMCVRotatedFeatureAlign"; }

  size_t GetInputTypeCount() const { return 2; }

  ONNXTensorElementDataType GetInputType(size_t) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }

  size_t GetOutputTypeCount() const { return 1; }

  ONNXTensorElementDataType GetOutputType(size_t) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }

  // force cpu
  const char* GetExecutionProviderType() const {
    return "CPUExecutionProvider";
  }
};
#endif  // ONNXRUNTIME_ROTATED_FEATURE_ALIGN_H
