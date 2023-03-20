// Copyright (c) OpenMMLab. All rights reserved
#ifndef CARAFE_CUDA_KERNEL_CUH
#define CARAFE_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

#ifdef MMCV_WITH_HIP
#define WARP_SIZE 64
#else
#define WARP_SIZE 32
#endif
#define THREADS_PER_PIXEL 32
#define MAX_SHARED_MEMORY 49152
#define MAX_SHARED_SCALAR_T 6144  // 49152 / 8 = 6144
#define MAXIMIZE_KERNEL_SIZE true
#define kTileDim 32
#define kBlockRows 8
#define FULL_MASK 0xffffffff

inline int divideUP(const int x, const int y) { return (((x) + (y)-1) / (y)); }

__device__ inline int Loc2Index(const int n, const int c, const int h,
                                const int w, const int channel_num,
                                const int height, const int width) {
  int index = w + (h + (c + n * channel_num) * height) * width;
  return index;
}
#ifndef MMCV_WITH_HIP
/* TODO: move this to a common place */
template <typename scalar_t>
__device__ inline scalar_t min(scalar_t a, scalar_t b) {
  return a < b ? a : b;
}

template <typename scalar_t>
__device__ inline scalar_t max(scalar_t a, scalar_t b) {
  return a > b ? a : b;
}
#endif
template <typename scalar_t>
__device__ __forceinline__ scalar_t warpReduceSum(scalar_t val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
#ifdef MMCV_WITH_HIP
    val += __shfl_down(val, offset);
#else
    val += __shfl_down_sync(FULL_MASK, val, offset);
#endif
  return val;
}

template <>
__device__ __forceinline__ phalf warpReduceSum(phalf val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
#ifdef MMCV_WITH_HIP
    __PHALF(val) += __shfl_down(val, offset);
#else
    __PHALF(val) +=
        __shfl_down_sync(FULL_MASK, static_cast<__half>(__PHALF(val)), offset);
#endif
  return val;
}

// Splits the original matrix into submatrices with size 32 * 32.
// Each block transposes one submatrix by loading it into shared memory.
// Reference https://devblogs.nvidia.com/efficient-matrix-transpose-cuda-cc/
template <typename scalar_t>
__global__ void BatchTranspose2DCUDAKernel(const int N, const int H,
                                           const int W, const int dh,
                                           const int dw,
                                           const scalar_t *__restrict__ X,
                                           scalar_t *__restrict__ Y) {
  __shared__ scalar_t tile[kTileDim][kTileDim + 1];
  const int n = blockIdx.x / (dh * dw);
  const int k = blockIdx.x % (dh * dw);
  const int r = k / dw;
  const int c = k % dw;
  const int offset = n * H * W;
  int x = c * kTileDim + threadIdx.x;
  int y = r * kTileDim + threadIdx.y;
  if (x < W) {
    for (int i = 0; threadIdx.y + i < kTileDim && y + i < H; i += kBlockRows) {
      tile[threadIdx.y + i][threadIdx.x] = X[offset + (y + i) * W + x];
    }
  }
  __syncthreads();
  x = r * kTileDim + threadIdx.x;
  y = c * kTileDim + threadIdx.y;
  if (x < H) {
    for (int i = 0; threadIdx.y + i < kTileDim && y + i < W; i += kBlockRows) {
      Y[offset + (y + i) * H + x] = tile[threadIdx.x][threadIdx.y + i];
    }
  }
}
template <typename scalar_t>
__global__ void CARAFEForward(
    const int num_kernels, const scalar_t *__restrict__ bottom_data,
    const scalar_t *__restrict__ bottom_masks, const int kernel_size,
    const int group_size, const int scale_factor, const int channels,
    const int down_height, const int down_width, const int height,
    const int width, const int mask_channels, scalar_t *__restrict__ top_data) {
#if MAXIMIZE_KERNEL_SIZE
  __shared__ float shared_mask[MAX_SHARED_SCALAR_T * 2];
#else
  __shared__ scalar_t shared_mask[MAX_SHARED_SCALAR_T];
#endif

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index > num_kernels - 1) {
    return;
  }
  const int pixel_id = threadIdx.x / THREADS_PER_PIXEL;
  const int split_id = threadIdx.x % THREADS_PER_PIXEL;
  index = index / THREADS_PER_PIXEL;
  const int pw = index % width;
  const int ph = (index / width) % height;
  const int n = index / width / height;

  const int down_pw = pw / scale_factor;
  const int down_ph = ph / scale_factor;

  const int start_w = down_pw - (kernel_size - 1) / 2;
  const int end_w = down_pw + (kernel_size - 1) / 2 + 1;
  const int start_h = down_ph - (kernel_size - 1) / 2;
  const int end_h = down_ph + (kernel_size - 1) / 2 + 1;
  for (int c = split_id; c < mask_channels; c += THREADS_PER_PIXEL) {
    int mask_index = Loc2Index(n, ph, pw, c, height, width, mask_channels);
    shared_mask[c * WARP_SIZE + pixel_id] = bottom_masks[mask_index];
  }
  __syncthreads();

  const int channels_per_group = ceilf(channels / (float)group_size);
#pragma unroll
  for (int c = split_id; c < channels; c += THREADS_PER_PIXEL) {
    int mask_group = c / channels_per_group;
    scalar_t output_val = 0;
#pragma unroll
    for (int iy = start_h; iy < end_h; iy++) {
#pragma unroll
      for (int ix = start_w; ix < end_w; ix++) {
        if (iy < 0 || iy > down_height - 1 || ix < 0 || ix > down_width - 1) {
          continue;
        }
        int mask_iy = iy - down_ph + (kernel_size - 1) / 2;
        int mask_ix = ix - down_pw + (kernel_size - 1) / 2;
        int mask_c =
            (mask_group * kernel_size + mask_iy) * kernel_size + mask_ix;
        int feat_index =
            Loc2Index(n, iy, ix, c, down_height, down_width, channels);

        output_val += bottom_data[feat_index] *
                      shared_mask[mask_c * WARP_SIZE + pixel_id];
      }
    }

    int top_index = Loc2Index(n, ph, pw, c, height, width, channels);
    top_data[top_index] = output_val;
  }
}

template <typename scalar_t>
__global__ void CARAFEBackward_Feature(
    const int num_kernels, const scalar_t *__restrict__ top_diff,
    const scalar_t *__restrict__ bottom_masks, const int kernel_size,
    const int group_size, const int scale_factor, const int channels,
    const int down_height, const int down_width, const int height,
    const int width, const int mask_channels,
    scalar_t *__restrict__ bottom_diff) {
#if MAXIMIZE_KERNEL_SIZE
  __shared__ float shared_mask[MAX_SHARED_SCALAR_T * 2];
#else
  __shared__ scalar_t shared_mask[MAX_SHARED_SCALAR_T];
#endif

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index > num_kernels - 1) {
    return;
  }

  const int pixel_id = threadIdx.x / THREADS_PER_PIXEL;
  const int split_id = threadIdx.x % THREADS_PER_PIXEL;
  // (n, c, ph, pw) is an element in the bottom_data
  index = index / THREADS_PER_PIXEL;
  const int pw = index % width;
  const int ph = (index / width) % height;
  const int n = index / width / height;

  const int start_w = pw - (kernel_size - 1) * scale_factor / 2;
  const int end_w = pw + (kernel_size - 1) * scale_factor / 2 + 1;
  const int start_h = ph - (kernel_size - 1) * scale_factor / 2;
  const int end_h = ph + (kernel_size - 1) * scale_factor / 2 + 1;
  for (int c = split_id; c < mask_channels; c += THREADS_PER_PIXEL) {
    const int mask_w = (c % kernel_size) * scale_factor;
    const int mask_h = (c / kernel_size % kernel_size) * scale_factor;
    const int mask_x = start_w + mask_w;
    const int mask_y = start_h + mask_h;
    if (mask_y < 0 || mask_y > height - 1 || mask_x < 0 || mask_x > width - 1) {
      shared_mask[c * WARP_SIZE + pixel_id] = 0;
      continue;
    }
    const int mask_group = c / (kernel_size * kernel_size);
    const int mask_c = (2 * mask_group + 1) * kernel_size * kernel_size - c - 1;
    int mask_index =
        Loc2Index(n, mask_c, mask_y, mask_x, mask_channels, height, width);
    shared_mask[c * WARP_SIZE + pixel_id] = bottom_masks[mask_index];
  }
  __syncthreads();
  const int channels_per_group = ceilf(channels / (float)group_size);
#pragma unroll
  for (int c = split_id; c < channels; c += THREADS_PER_PIXEL) {
    int mask_group = c / channels_per_group;
    int top_index = Loc2Index(n, ph, pw, c, height, width, channels);
    scalar_t output_val = 0;
#pragma unroll
    for (int iy = start_h; iy < end_h; iy += scale_factor) {
#pragma unroll
      for (int ix = start_w; ix < end_w; ix += scale_factor) {
        if (iy < 0 || iy > height - 1 || ix < 0 || ix > width - 1) {
          continue;
        }
        int mask_iy =
            (iy - ph + (kernel_size - 1) * scale_factor / 2) / scale_factor;
        int mask_ix =
            (ix - pw + (kernel_size - 1) * scale_factor / 2) / scale_factor;
        int mask_c =
            (mask_group * kernel_size + mask_iy) * kernel_size + mask_ix;
        int feat_index = Loc2Index(n, iy, ix, c, height, width, channels);
        output_val +=
            shared_mask[mask_c * WARP_SIZE + pixel_id] * top_diff[feat_index];
      }
    }
    bottom_diff[top_index] = output_val;
  }
}

template <typename scalar_t>
__global__ void FeatureSum(const int num_kernels,
                           const scalar_t *__restrict__ input_data,
                           const int scale_factor, const int channels,
                           const int height, const int width,
                           scalar_t *__restrict__ output_data) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index > num_kernels - 1) {
    return;
  }
  const int split_id = threadIdx.x % THREADS_PER_PIXEL;
  index = index / THREADS_PER_PIXEL;
  const int pw = index % width;
  const int ph = (index / width) % height;
  const int n = index / width / height;
  for (int c = split_id; c < channels; c += THREADS_PER_PIXEL) {
    scalar_t output_val = 0;
    for (int iy = ph * scale_factor; iy < (ph + 1) * scale_factor; iy++) {
      for (int ix = pw * scale_factor; ix < (pw + 1) * scale_factor; ix++) {
        int input_id = Loc2Index(n, iy, ix, c, height * scale_factor,
                                 width * scale_factor, channels);
        output_val += input_data[input_id];
      }
    }
    const int output_id = Loc2Index(n, ph, pw, c, height, width, channels);
    output_data[output_id] = output_val;
  }
}

template <typename scalar_t>
__global__ void CARAFEBackward_Mask(const int num_kernels,
                                    const scalar_t *__restrict__ top_diff,
                                    const scalar_t *__restrict__ bottom_data,
                                    const int kernel_size, const int group_size,
                                    const int scale_factor, const int channels,
                                    const int down_height, const int down_width,
                                    const int height, const int width,
                                    const int mask_channels,
                                    scalar_t *__restrict__ mask_diff) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index > num_kernels - 1) {
    return;
  }

  const int lane_id = index % WARP_SIZE;
  index = index / WARP_SIZE;
  const int mask_c = index % mask_channels;
  // (n, c, ph, pw) is an element in the bottom_data
  index = index / mask_channels;
  const int pw = index % width;
  const int ph = (index / width) % height;
  const int n = index / width / height;

  const int down_pw = pw / scale_factor;
  const int down_ph = ph / scale_factor;

  const int mask_group = mask_c / (kernel_size * kernel_size);
  const int mask_loc = mask_c % (kernel_size * kernel_size);

  const int offset_x = mask_loc % kernel_size - (kernel_size - 1) / 2;
  const int offset_y =
      mask_loc / kernel_size % kernel_size - (kernel_size - 1) / 2;

  const int down_x = down_pw + offset_x;
  const int down_y = down_ph + offset_y;

  scalar_t output_val = 0;

  if (down_y >= 0 && down_y <= down_height - 1 && down_x >= 0 &&
      down_x <= down_width - 1) {
    const int channels_per_mask = ceilf(channels / (float)group_size);
    const int start = channels_per_mask * mask_group;
    const int end = min(channels_per_mask * (mask_group + 1), channels);
    for (int c = start + lane_id; c < end; c += WARP_SIZE) {
      int bottom_id =
          Loc2Index(n, down_y, down_x, c, down_height, down_width, channels);
      int top_id = Loc2Index(n, ph, pw, c, height, width, channels);
      output_val += top_diff[top_id] * bottom_data[bottom_id];
    }
  }
#ifdef MMCV_WITH_HIP
  __syncthreads();
#else
  __syncwarp();
#endif
  output_val = warpReduceSum(output_val);
  if (lane_id == 0) {
    const int mask_id =
        Loc2Index(n, ph, pw, mask_c, height, width, mask_channels);
    mask_diff[mask_id] = output_val;
  }
}

#endif  // CARAFE_CUDA_KERNEL_CUH
