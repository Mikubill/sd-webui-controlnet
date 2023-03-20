// Copyright (c) OpenMMLab. All rights reserved
#ifndef ORT_MMCV_UTILS_H
#define ORT_MMCV_UTILS_H
#include <onnxruntime_cxx_api.h>

#include <vector>

struct OrtTensorDimensions : std::vector<int64_t> {
  OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue* value) {
    OrtTensorTypeAndShapeInfo* info = ort.GetTensorTypeAndShape(value);
    std::vector<int64_t>::operator=(ort.GetTensorShape(info));
    ort.ReleaseTensorTypeAndShapeInfo(info);
  }
};
#endif  // ORT_MMCV_UTILS_H
