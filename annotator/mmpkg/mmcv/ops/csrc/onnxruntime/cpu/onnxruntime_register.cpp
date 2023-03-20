// Copyright (c) OpenMMLab. All rights reserved
#include "onnxruntime_register.h"

#include "corner_pool.h"
#include "deform_conv.h"
#include "grid_sample.h"
#include "modulated_deform_conv.h"
#include "nms.h"
#include "ort_mmcv_utils.h"
#include "reduce_ops.h"
#include "roi_align.h"
#include "roi_align_rotated.h"
#include "rotated_feature_align.h"
#include "soft_nms.h"

const char *c_MMCVOpDomain = "mmcv";
SoftNmsOp c_SoftNmsOp;
NmsOp c_NmsOp;
MMCVRoiAlignCustomOp c_MMCVRoiAlignCustomOp;
MMCVRoIAlignRotatedCustomOp c_MMCVRoIAlignRotatedCustomOp;
MMCVRotatedFeatureAlignCustomOp c_MMCVRotatedFeatureAlignCustomOp;
GridSampleOp c_GridSampleOp;
MMCVCumMaxCustomOp c_MMCVCumMaxCustomOp;
MMCVCumMinCustomOp c_MMCVCumMinCustomOp;
MMCVCornerPoolCustomOp c_MMCVCornerPoolCustomOp;
MMCVModulatedDeformConvOp c_MMCVModulatedDeformConvOp;
MMCVDeformConvOp c_MMCVDeformConvOp;

OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options,
                                          const OrtApiBase *api) {
  OrtCustomOpDomain *domain = nullptr;
  const OrtApi *ortApi = api->GetApi(ORT_API_VERSION);

  if (auto status = ortApi->CreateCustomOpDomain(c_MMCVOpDomain, &domain)) {
    return status;
  }

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_SoftNmsOp)) {
    return status;
  }

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_NmsOp)) {
    return status;
  }

  if (auto status =
          ortApi->CustomOpDomain_Add(domain, &c_MMCVRoiAlignCustomOp)) {
    return status;
  }

  if (auto status =
          ortApi->CustomOpDomain_Add(domain, &c_MMCVRoIAlignRotatedCustomOp)) {
    return status;
  }

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_GridSampleOp)) {
    return status;
  }

  if (auto status =
          ortApi->CustomOpDomain_Add(domain, &c_MMCVCornerPoolCustomOp)) {
    return status;
  }

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_MMCVCumMaxCustomOp)) {
    return status;
  }

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_MMCVCumMinCustomOp)) {
    return status;
  }

  if (auto status =
          ortApi->CustomOpDomain_Add(domain, &c_MMCVModulatedDeformConvOp)) {
    return status;
  }

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_MMCVDeformConvOp)) {
    return status;
  }

  if (auto status = ortApi->CustomOpDomain_Add(
          domain, &c_MMCVRotatedFeatureAlignCustomOp)) {
    return status;
  }

  return ortApi->AddCustomOpDomain(options, domain);
}
