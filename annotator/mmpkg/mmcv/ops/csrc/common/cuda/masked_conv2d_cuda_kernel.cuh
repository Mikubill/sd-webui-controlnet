// Copyright (c) OpenMMLab. All rights reserved
#ifndef MASKED_CONV2D_CUDA_KERNEL_CUH
#define MASKED_CONV2D_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

template <typename scalar_t>
__global__ void MaskedIm2colForward(const int n, const scalar_t *data_im,
                                    const int height, const int width,
                                    const int kernel_h, const int kernel_w,
                                    const int pad_h, const int pad_w,
                                    const int64_t *mask_h_idx,
                                    const int64_t *mask_w_idx,
                                    const int mask_cnt, scalar_t *data_col) {
  // mask_cnt * channels
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int m_index = index % mask_cnt;
    const int h_col = mask_h_idx[m_index];
    const int w_col = mask_w_idx[m_index];
    const int c_im = index / mask_cnt;
    const int c_col = c_im * kernel_h * kernel_w;
    const int h_offset = h_col - pad_h;
    const int w_offset = w_col - pad_w;
    scalar_t *data_col_ptr = data_col + c_col * mask_cnt + m_index;
    for (int i = 0; i < kernel_h; ++i) {
      int h_im = h_offset + i;
      for (int j = 0; j < kernel_w; ++j) {
        int w_im = w_offset + j;
        if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
          *data_col_ptr =
              (scalar_t)data_im[(c_im * height + h_im) * width + w_im];
        } else {
          *data_col_ptr = 0.0;
        }
        data_col_ptr += mask_cnt;
      }
    }
  }
}

template <typename scalar_t>
__global__ void MaskedCol2imForward(const int n, const scalar_t *data_col,
                                    const int height, const int width,
                                    const int channels,
                                    const int64_t *mask_h_idx,
                                    const int64_t *mask_w_idx,
                                    const int mask_cnt, scalar_t *data_im) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int m_index = index % mask_cnt;
    const int h_im = mask_h_idx[m_index];
    const int w_im = mask_w_idx[m_index];
    const int c_im = index / mask_cnt;
    // compute the start and end of the output
    data_im[(c_im * height + h_im) * width + w_im] = data_col[index];
  }
}

#endif  // MASKED_CONV2D_CUDA_KERNEL_CUH
