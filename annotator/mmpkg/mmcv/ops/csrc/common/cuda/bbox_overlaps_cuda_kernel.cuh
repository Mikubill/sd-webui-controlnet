// Copyright (c) OpenMMLab. All rights reserved
#ifndef BBOX_OVERLAPS_CUDA_KERNEL_CUH
#define BBOX_OVERLAPS_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

template <typename T>
__device__ __forceinline__ void load_bbox(const T* bbox, const int base, T& x1,
                                          T& y1, T& x2, T& y2) {
  x1 = bbox[base];
  y1 = bbox[base + 1];
  x2 = bbox[base + 2];
  y2 = bbox[base + 3];
}

template <>
__device__ __forceinline__ void load_bbox<float>(const float* bbox,
                                                 const int base, float& x1,
                                                 float& y1, float& x2,
                                                 float& y2) {
  const float4 bbox_offset = reinterpret_cast<const float4*>(bbox + base)[0];
  x1 = bbox_offset.x;
  y1 = bbox_offset.y;
  x2 = bbox_offset.z;
  y2 = bbox_offset.w;
}

template <typename T>
__global__ void bbox_overlaps_cuda_kernel(const T* bbox1, const T* bbox2,
                                          T* ious, const int num_bbox1,
                                          const int num_bbox2, const int mode,
                                          const bool aligned,
                                          const int offset) {
  if (aligned) {
    CUDA_1D_KERNEL_LOOP(index, num_bbox1) {
      const int b1 = index;
      const int b2 = index;

      const int base1 = b1 << 2;  // b1 * 4
      T b1_x1, b1_y1, b1_x2, b1_y2;
      load_bbox<T>(bbox1, base1, b1_x1, b1_y1, b1_x2, b1_y2);
      const T b1_area = (b1_x2 - b1_x1 + offset) * (b1_y2 - b1_y1 + offset);

      const int base2 = b2 << 2;  // b2 * 4
      T b2_x1, b2_y1, b2_x2, b2_y2;
      load_bbox<T>(bbox2, base2, b2_x1, b2_y1, b2_x2, b2_y2);
      const T b2_area = (b2_x2 - b2_x1 + offset) * (b2_y2 - b2_y1 + offset);

      const T left = fmaxf(b1_x1, b2_x1), right = fminf(b1_x2, b2_x2);
      const T top = fmaxf(b1_y1, b2_y1), bottom = fminf(b1_y2, b2_y2);
      const T width = fmaxf(right - left + offset, 0.f);
      const T height = fmaxf(bottom - top + offset, 0.f);
      const T interS = width * height;

      const T baseS =
          fmaxf(mode == 0 ? b1_area + b2_area - interS : b1_area, T(offset));
      ious[index] = interS / baseS;
    }
  } else {
    CUDA_1D_KERNEL_LOOP(index, num_bbox1 * num_bbox2) {
      const int b1 = index / num_bbox2;
      const int b2 = index % num_bbox2;

      const int base1 = b1 << 2;  // b1 * 4
      T b1_x1, b1_y1, b1_x2, b1_y2;
      load_bbox<T>(bbox1, base1, b1_x1, b1_y1, b1_x2, b1_y2);
      const T b1_area = (b1_x2 - b1_x1 + offset) * (b1_y2 - b1_y1 + offset);

      const int base2 = b2 << 2;  // b2 * 4
      T b2_x1, b2_y1, b2_x2, b2_y2;
      load_bbox<T>(bbox2, base2, b2_x1, b2_y1, b2_x2, b2_y2);
      const T b2_area = (b2_x2 - b2_x1 + offset) * (b2_y2 - b2_y1 + offset);

      const T left = fmaxf(b1_x1, b2_x1), right = fminf(b1_x2, b2_x2);
      const T top = fmaxf(b1_y1, b2_y1), bottom = fminf(b1_y2, b2_y2);
      const T width = fmaxf(right - left + offset, 0.f);
      const T height = fmaxf(bottom - top + offset, 0.f);
      const T interS = width * height;

      const T baseS =
          fmaxf(mode == 0 ? b1_area + b2_area - interS : b1_area, T(offset));
      ious[index] = interS / baseS;
    }
  }
}

#if __CUDA_ARCH__ >= 530
__device__ __forceinline__ __half __half_area(const __half x1, const __half y1,
                                              const __half x2, const __half y2,
                                              const __half offset) {
  const __half half_w = __hadd(__hsub(x2, x1), offset);
  const __half half_h = __hadd(__hsub(y2, y1), offset);
  return __hmul(half_w, half_h);
}

__device__ __forceinline__ __half __half_max(const __half a, const __half b) {
  return __hge(a, b) ? a : b;
}

__device__ __forceinline__ __half __half_min(const __half a, const __half b) {
  return __hle(a, b) ? a : b;
}

// fp16 won't provide much increase when aligned==true. It is useful when
// aligned==false, which would give you ~40% bonus.
__device__ void bbox_overlaps_cuda_kernel_half(
    const __half* bbox1, const __half* bbox2, __half* ious, const int num_bbox1,
    const int num_bbox2, const int mode, const bool aligned, const int offset) {
  const int num_output = aligned ? num_bbox1 : num_bbox1 * num_bbox2;
  const __half h_offset = __int2half_rn(offset);
  CUDA_1D_KERNEL_LOOP(index, num_output) {
    const int b1 = aligned ? index : index / num_bbox2;
    const int b2 = aligned ? index : index % num_bbox2;

    const int base1 = b1 << 2;
    __half b1_x1, b1_y1, b1_x2, b1_y2;
    load_bbox<__half>(bbox1, base1, b1_x1, b1_y1, b1_x2, b1_y2);
    const __half b1_area = __half_area(b1_x1, b1_y1, b1_x2, b1_y2, h_offset);

    const int base2 = b2 << 2;
    __half b2_x1, b2_y1, b2_x2, b2_y2;
    load_bbox<__half>(bbox2, base2, b2_x1, b2_y1, b2_x2, b2_y2);
    const __half b2_area = __half_area(b2_x1, b2_y1, b2_x2, b2_y2, h_offset);

    const __half left = __half_max(b1_x1, b2_x1),
                 right = __half_min(b1_x2, b2_x2);
    const __half top = __half_max(b1_y1, b2_y1),
                 bottom = __half_min(b1_y2, b2_y2);
    const __half width =
        __half_max(__hadd(__hsub(right, left), h_offset), __float2half(0.f));
    const __half height =
        __half_max(__hadd(__hsub(bottom, top), h_offset), __float2half(0.f));
    const __half interS = __hmul(width, height);

    const __half baseS = __half_max(
        mode == 0 ? __hsub(__hadd(b1_area, b2_area), interS) : b1_area,
        h_offset);
    ious[index] = __hdiv(interS, baseS);
  }
}
#endif  // __CUDA_ARCH__ >= 530

#endif  // BBOX_OVERLAPS_CUDA_KERNEL_CUH
