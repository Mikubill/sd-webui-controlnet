// Copyright (c) OpenMMLab. All rights reserved
#ifndef PSAMASK_PYTORCH_H
#define PSAMASK_PYTORCH_H
#include <torch/extension.h>
using namespace at;

#ifdef MMCV_WITH_CUDA
void psamask_forward_cuda(const int psa_type, const Tensor input, Tensor output,
                          const int num_, const int h_feature,
                          const int w_feature, const int h_mask,
                          const int w_mask, const int half_h_mask,
                          const int half_w_mask);

void psamask_backward_cuda(const int psa_type, const Tensor grad_output,
                           Tensor grad_input, const int num_,
                           const int h_feature, const int w_feature,
                           const int h_mask, const int w_mask,
                           const int half_h_mask, const int half_w_mask);
#endif
void psamask_forward_cpu(const int psa_type, const Tensor input, Tensor output,
                         const int num_, const int h_feature,
                         const int w_feature, const int h_mask,
                         const int w_mask, const int half_h_mask,
                         const int half_w_mask);

void psamask_backward_cpu(const int psa_type, const Tensor grad_output,
                          Tensor grad_input, const int num_,
                          const int h_feature, const int w_feature,
                          const int h_mask, const int w_mask,
                          const int half_h_mask, const int half_w_mask);
#endif  // PSAMASK_PYTORCH_H
