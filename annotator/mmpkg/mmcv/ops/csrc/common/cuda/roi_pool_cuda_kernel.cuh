// Copyright (c) OpenMMLab. All rights reserved
#ifndef ROI_POOL_CUDA_KERNEL_CUH
#define ROI_POOL_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

template <typename T>
__global__ void roi_pool_forward_cuda_kernel(
    const int nthreads, const T* input, const T* rois, T* output, int* argmax,
    const int pooled_height, const int pooled_width, const T spatial_scale,
    const int channels, const int height, const int width) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];
    // calculate the roi region on feature maps
    T roi_x1 = offset_rois[1] * spatial_scale;
    T roi_y1 = offset_rois[2] * spatial_scale;
    T roi_x2 = (offset_rois[3] + 1) * spatial_scale;
    T roi_y2 = (offset_rois[4] + 1) * spatial_scale;

    // force malformed rois to be 1x1
    T roi_w = roi_x2 - roi_x1;
    T roi_h = roi_y2 - roi_y1;
    if (roi_w <= 0 || roi_h <= 0) continue;

    T bin_size_w = roi_w / static_cast<T>(pooled_width);
    T bin_size_h = roi_h / static_cast<T>(pooled_height);

    // the corresponding bin region
    int bin_x1 = floorf(static_cast<T>(pw) * bin_size_w + roi_x1);
    int bin_y1 = floorf(static_cast<T>(ph) * bin_size_h + roi_y1);
    int bin_x2 = ceilf(static_cast<T>(pw + 1) * bin_size_w + roi_x1);
    int bin_y2 = ceilf(static_cast<T>(ph + 1) * bin_size_h + roi_y1);

    // add roi offsets and clip to input boundaries
    bin_x1 = min(max(bin_x1, 0), width);
    bin_y1 = min(max(bin_y1, 0), height);
    bin_x2 = min(max(bin_x2, 0), width);
    bin_y2 = min(max(bin_y2, 0), height);
    bool is_empty = (bin_y2 <= bin_y1) || (bin_x2 <= bin_x1);

    const T* offset_input =
        input + (roi_batch_ind * channels + c) * height * width;
    // Define an empty pooling region to be zero
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    T max_val = is_empty ? 0 : -FLT_MAX;
    int max_idx = -1;
    for (int h = bin_y1; h < bin_y2; ++h) {
      for (int w = bin_x1; w < bin_x2; ++w) {
        int offset = h * width + w;
        if (offset_input[offset] > max_val) {
          max_val = offset_input[offset];
          max_idx = offset;
        }
      }
    }
    output[index] = max_val;
    if (argmax != NULL) argmax[index] = max_idx;
  }
}

template <typename T>
__global__ void roi_pool_backward_cuda_kernel(
    const int nthreads, const T* grad_output, const T* rois, const int* argmax,
    T* grad_input, const int pooled_height, const int pooled_width,
    const int channels, const int height, const int width) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c) is an element in the pooled output
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    int roi_batch_ind = rois[n * 5];
    T* grad_input_offset =
        grad_input + ((roi_batch_ind * channels + c) * height * width);
    int argmax_index = argmax[index];

    if (argmax_index != -1) {
      atomicAdd(grad_input_offset + argmax_index, grad_output[index]);
    }
  }
}

#endif  // ROI_POOL_CUDA_KERNEL_CUH
