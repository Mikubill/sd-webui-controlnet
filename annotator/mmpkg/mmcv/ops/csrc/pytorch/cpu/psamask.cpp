// Copyright (c) OpenMMLab. All rights reserved
// Modified from
// https://github.com/hszhao/semseg/blob/master/lib/psa/src
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

#ifndef min
#define min(a, b) (((a) < (b)) ? (a) : (b))
#endif
#ifndef max
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif

void psamask_collect_forward(const int num_, const int h_feature,
                             const int w_feature, const int h_mask,
                             const int w_mask, const int half_h_mask,
                             const int half_w_mask, const Tensor mask_data,
                             Tensor buffer_data) {
  for (int n = 0; n < num_; n++) {
    for (int h = 0; h < h_feature; h++) {
      for (int w = 0; w < w_feature; w++) {
        // effective mask region : [hstart, hend) x [wstart, wend) with
        // mask-indexed
        const int hstart = max(0, half_h_mask - h);
        const int hend = min(h_mask, h_feature + half_h_mask - h);
        const int wstart = max(0, half_w_mask - w);
        const int wend = min(w_mask, w_feature + half_w_mask - w);
        // (hidx,                    widx                   ) with mask-indexed
        // (hidx + h - half_h_mask, widx + w - half_w_mask) with
        // feature-indexed
        for (int hidx = hstart; hidx < hend; hidx++) {
          for (int widx = wstart; widx < wend; widx++) {
            buffer_data.view({-1})[(n * h_feature * w_feature +
                                    (hidx + h - half_h_mask) * w_feature +
                                    (widx + w - half_w_mask)) *
                                       h_feature * w_feature +
                                   h * w_feature + w] =
                mask_data.view(
                    {-1})[((n * h_mask * w_mask + hidx * w_mask + widx) *
                               h_feature +
                           h) *
                              w_feature +
                          w];
          }
        }
      }
    }
  }
}

void psamask_distribute_forward(const int num_, const int h_feature,
                                const int w_feature, const int h_mask,
                                const int w_mask, const int half_h_mask,
                                const int half_w_mask, const Tensor mask_data,
                                Tensor buffer_data) {
  for (int n = 0; n < num_; n++) {
    for (int h = 0; h < h_feature; h++) {
      for (int w = 0; w < w_feature; w++) {
        // effective mask region : [hstart, hend) x [wstart, wend) with
        // mask-indexed
        const int hstart = max(0, half_h_mask - h);
        const int hend = min(h_mask, h_feature + half_h_mask - h);
        const int wstart = max(0, half_w_mask - w);
        const int wend = min(w_mask, w_feature + half_w_mask - w);
        // (hidx,                    widx                   ) with mask-indexed
        // (hidx + h - half_h_mask, widx + w - half_w_mask) with
        // feature-indexed
        for (int hidx = hstart; hidx < hend; hidx++) {
          for (int widx = wstart; widx < wend; widx++) {
            buffer_data.view(
                {-1})[(n * h_feature * w_feature + h * w_feature + w) *
                          h_feature * w_feature +
                      (hidx + h - half_h_mask) * w_feature +
                      (widx + w - half_w_mask)] =
                mask_data.view(
                    {-1})[((n * h_mask * w_mask + hidx * w_mask + widx) *
                               h_feature +
                           h) *
                              w_feature +
                          w];
          }
        }
      }
    }
  }
}

void psamask_collect_backward(const int num_, const int h_feature,
                              const int w_feature, const int h_mask,
                              const int w_mask, const int half_h_mask,
                              const int half_w_mask, const Tensor buffer_diff,
                              Tensor mask_diff) {
  for (int n = 0; n < num_; n++) {
    for (int h = 0; h < h_feature; h++) {
      for (int w = 0; w < w_feature; w++) {
        // effective mask region : [hstart, hend) x [wstart, wend) with
        // mask-indexed
        const int hstart = max(0, half_h_mask - h);
        const int hend = min(h_mask, h_feature + half_h_mask - h);
        const int wstart = max(0, half_w_mask - w);
        const int wend = min(w_mask, w_feature + half_w_mask - w);
        // (hidx,                    widx                   ) with mask-indexed
        // (hidx + h - half_h_mask, widx + w - half_w_mask) with
        // feature-indexed
        for (int hidx = hstart; hidx < hend; hidx++) {
          for (int widx = wstart; widx < wend; widx++) {
            mask_diff.view({-1})[((n * h_mask * w_mask + hidx * w_mask + widx) *
                                      h_feature +
                                  h) *
                                     w_feature +
                                 w] =
                buffer_diff.view({-1})[(n * h_feature * w_feature +
                                        (hidx + h - half_h_mask) * w_feature +
                                        (widx + w - half_w_mask)) *
                                           h_feature * w_feature +
                                       h * w_feature + w];
          }
        }
      }
    }
  }
}

void psamask_distribute_backward(const int num_, const int h_feature,
                                 const int w_feature, const int h_mask,
                                 const int w_mask, const int half_h_mask,
                                 const int half_w_mask,
                                 const Tensor buffer_diff, Tensor mask_diff) {
  for (int n = 0; n < num_; n++) {
    for (int h = 0; h < h_feature; h++) {
      for (int w = 0; w < w_feature; w++) {
        // effective mask region : [hstart, hend) x [wstart, wend) with
        // mask-indexed
        const int hstart = max(0, half_h_mask - h);
        const int hend = min(h_mask, h_feature + half_h_mask - h);
        const int wstart = max(0, half_w_mask - w);
        const int wend = min(w_mask, w_feature + half_w_mask - w);
        // (hidx,                    widx                   ) with mask-indexed
        // (hidx + h - half_h_mask, widx + w - half_w_mask) with
        // feature-indexed
        for (int hidx = hstart; hidx < hend; hidx++) {
          for (int widx = wstart; widx < wend; widx++) {
            mask_diff.view({-1})[((n * h_mask * w_mask + hidx * w_mask + widx) *
                                      h_feature +
                                  h) *
                                     w_feature +
                                 w] =
                buffer_diff.view(
                    {-1})[(n * h_feature * w_feature + h * w_feature + w) *
                              h_feature * w_feature +
                          (hidx + h - half_h_mask) * w_feature +
                          (widx + w - half_w_mask)];
          }
        }
      }
    }
  }
}

void psamask_forward_cpu(const int psa_type, const Tensor input, Tensor output,
                         const int num_, const int h_feature,
                         const int w_feature, const int h_mask,
                         const int w_mask, const int half_h_mask,
                         const int half_w_mask) {
  if (psa_type == 0)
    psamask_collect_forward(num_, h_feature, w_feature, h_mask, w_mask,
                            half_h_mask, half_w_mask, input, output);
  else
    psamask_distribute_forward(num_, h_feature, w_feature, h_mask, w_mask,
                               half_h_mask, half_w_mask, input, output);
}

void psamask_backward_cpu(const int psa_type, const Tensor grad_output,
                          Tensor grad_input, const int num_,
                          const int h_feature, const int w_feature,
                          const int h_mask, const int w_mask,
                          const int half_h_mask, const int half_w_mask) {
  if (psa_type == 0)
    psamask_collect_backward(num_, h_feature, w_feature, h_mask, w_mask,
                             half_h_mask, half_w_mask, grad_output, grad_input);
  else
    psamask_distribute_backward(num_, h_feature, w_feature, h_mask, w_mask,
                                half_h_mask, half_w_mask, grad_output,
                                grad_input);
}

void psamask_forward_impl(const int psa_type, const Tensor input, Tensor output,
                          const int num_, const int h_feature,
                          const int w_feature, const int h_mask,
                          const int w_mask, const int half_h_mask,
                          const int half_w_mask);

void psamask_backward_impl(const int psa_type, const Tensor grad_output,
                           Tensor grad_input, const int num_,
                           const int h_feature, const int w_feature,
                           const int h_mask, const int w_mask,
                           const int half_h_mask, const int half_w_mask);
REGISTER_DEVICE_IMPL(psamask_forward_impl, CPU, psamask_forward_cpu);
REGISTER_DEVICE_IMPL(psamask_backward_impl, CPU, psamask_backward_cpu);
