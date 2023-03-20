// Copyright (c) OpenMMLab. All rights reserved
// Modified from
// https://github.com/hszhao/semseg/blob/master/lib/psa/src
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void psamask_forward_impl(const int psa_type, const Tensor input, Tensor output,
                          const int num_, const int h_feature,
                          const int w_feature, const int h_mask,
                          const int w_mask, const int half_h_mask,
                          const int half_w_mask) {
  DISPATCH_DEVICE_IMPL(psamask_forward_impl, psa_type, input, output, num_,
                       h_feature, w_feature, h_mask, w_mask, half_h_mask,
                       half_w_mask);
}

void psamask_backward_impl(const int psa_type, const Tensor grad_output,
                           Tensor grad_input, const int num_,
                           const int h_feature, const int w_feature,
                           const int h_mask, const int w_mask,
                           const int half_h_mask, const int half_w_mask) {
  DISPATCH_DEVICE_IMPL(psamask_backward_impl, psa_type, grad_output, grad_input,
                       num_, h_feature, w_feature, h_mask, w_mask, half_h_mask,
                       half_w_mask);
}

void psamask_forward(const Tensor input, Tensor output, const int psa_type,
                     const int num_, const int h_feature, const int w_feature,
                     const int h_mask, const int w_mask, const int half_h_mask,
                     const int half_w_mask) {
  psamask_forward_impl(psa_type, input, output, num_, h_feature, w_feature,
                       h_mask, w_mask, half_h_mask, half_w_mask);
}

void psamask_backward(Tensor grad_output, const Tensor grad_input,
                      const int psa_type, const int num_, const int h_feature,
                      const int w_feature, const int h_mask, const int w_mask,
                      const int half_h_mask, const int half_w_mask) {
  psamask_backward_impl(psa_type, grad_output, grad_input, num_, h_feature,
                        w_feature, h_mask, w_mask, half_h_mask, half_w_mask);
}
