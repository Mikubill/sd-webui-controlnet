// Copyright (c) OpenMMLab. All rights reserved
#ifndef PSAMASK_CUDA_KERNEL_CUH
#define PSAMASK_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

// CUDA: grid stride looping
#ifndef CUDA_KERNEL_LOOP
#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)
#endif

template <typename T>
__global__ void psamask_collect_forward_cuda(
    const int nthreads, const int h_feature, const int w_feature,
    const int h_mask, const int w_mask, const int half_h_mask,
    const int half_w_mask, const T* mask_data, T* buffer_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % w_feature;
    const int h = (index / w_feature) % h_feature;
    const int n = index / w_feature / h_feature;
    // effective mask region : [hstart, hend) x [wstart, wend) with mask-indexed
    const int hstart = max(0, half_h_mask - h);
    const int hend = min(h_mask, h_feature + half_h_mask - h);
    const int wstart = max(0, half_w_mask - w);
    const int wend = min(w_mask, w_feature + half_w_mask - w);
    // (hidx,                    widx                   ) with mask-indexed
    // (hidx + h - half_h_mask, widx + w - half_w_mask) with feature-indexed
    for (int hidx = hstart; hidx < hend; hidx++) {
      for (int widx = wstart; widx < wend; widx++) {
        buffer_data[(n * h_feature * w_feature +
                     (hidx + h - half_h_mask) * w_feature +
                     (widx + w - half_w_mask)) *
                        h_feature * w_feature +
                    h * w_feature + w] = mask_data
            [((n * h_mask * w_mask + hidx * w_mask + widx) * h_feature + h) *
                 w_feature +
             w];
      }
    }
  }
}

template <typename T>
__global__ void psamask_distribute_forward_cuda(
    const int nthreads, const int h_feature, const int w_feature,
    const int h_mask, const int w_mask, const int half_h_mask,
    const int half_w_mask, const T* mask_data, T* buffer_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % w_feature;
    const int h = (index / w_feature) % h_feature;
    const int n = index / w_feature / h_feature;
    // effective mask region : [hstart, hend) x [wstart, wend) with mask-indexed
    const int hstart = max(0, half_h_mask - h);
    const int hend = min(h_mask, h_feature + half_h_mask - h);
    const int wstart = max(0, half_w_mask - w);
    const int wend = min(w_mask, w_feature + half_w_mask - w);
    // (hidx,                    widx                   ) with mask-indexed
    // (hidx + h - half_h_mask, widx + w - half_w_mask) with feature-indexed
    for (int hidx = hstart; hidx < hend; hidx++) {
      for (int widx = wstart; widx < wend; widx++) {
        buffer_data[(n * h_feature * w_feature + h * w_feature + w) *
                        h_feature * w_feature +
                    (hidx + h - half_h_mask) * w_feature +
                    (widx + w - half_w_mask)] = mask_data
            [((n * h_mask * w_mask + hidx * w_mask + widx) * h_feature + h) *
                 w_feature +
             w];
      }
    }
  }
}

template <typename T>
__global__ void psamask_collect_backward_cuda(
    const int nthreads, const int h_feature, const int w_feature,
    const int h_mask, const int w_mask, const int half_h_mask,
    const int half_w_mask, const T* buffer_diff, T* mask_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % w_feature;
    const int h = (index / w_feature) % h_feature;
    const int n = index / w_feature / h_feature;
    // effective mask region : [hstart, hend) x [wstart, wend) with mask-indexed
    const int hstart = max(0, half_h_mask - h);
    const int hend = min(h_mask, h_feature + half_h_mask - h);
    const int wstart = max(0, half_w_mask - w);
    const int wend = min(w_mask, w_feature + half_w_mask - w);
    // (hidx,                    widx                   ) with mask-indexed
    // (hidx + h - half_h_mask, widx + w - half_w_mask) with feature-indexed
    for (int hidx = hstart; hidx < hend; hidx++) {
      for (int widx = wstart; widx < wend; widx++) {
        mask_diff[((n * h_mask * w_mask + hidx * w_mask + widx) * h_feature +
                   h) *
                      w_feature +
                  w] = buffer_diff[(n * h_feature * w_feature +
                                    (hidx + h - half_h_mask) * w_feature +
                                    (widx + w - half_w_mask)) *
                                       h_feature * w_feature +
                                   h * w_feature + w];
      }
    }
  }
}

template <typename T>
__global__ void psamask_distribute_backward_cuda(
    const int nthreads, const int h_feature, const int w_feature,
    const int h_mask, const int w_mask, const int half_h_mask,
    const int half_w_mask, const T* buffer_diff, T* mask_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % w_feature;
    const int h = (index / w_feature) % h_feature;
    const int n = index / w_feature / h_feature;
    // effective mask region : [hstart, hend) x [wstart, wend) with mask-indexed
    const int hstart = max(0, half_h_mask - h);
    const int hend = min(h_mask, h_feature + half_h_mask - h);
    const int wstart = max(0, half_w_mask - w);
    const int wend = min(w_mask, w_feature + half_w_mask - w);
    // (hidx,                    widx                   ) with mask-indexed
    // (hidx + h - half_h_mask, widx + w - half_w_mask) with feature-indexed
    for (int hidx = hstart; hidx < hend; hidx++) {
      for (int widx = wstart; widx < wend; widx++) {
        mask_diff[((n * h_mask * w_mask + hidx * w_mask + widx) * h_feature +
                   h) *
                      w_feature +
                  w] =
            buffer_diff[(n * h_feature * w_feature + h * w_feature + w) *
                            h_feature * w_feature +
                        (hidx + h - half_h_mask) * w_feature +
                        (widx + w - half_w_mask)];
      }
    }
  }
}

#endif  // PSAMASK_CUDA_KERNEL_CUH
