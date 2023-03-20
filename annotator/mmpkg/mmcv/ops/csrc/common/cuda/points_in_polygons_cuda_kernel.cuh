// Copyright (c) OpenMMLab. All rights reserved
#ifndef POINTS_IN_POLYGONS_CUDA_KERNEL_CUH
#define POINTS_IN_POLYGONS_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

struct point {
  float x, y;
};

template <typename scalar_t>
__global__ void points_in_polygons_forward_cuda_kernel(
    const int nthreads, const scalar_t *vertex1, const scalar_t *vertex2,
    const int rows, const int cols, scalar_t *inside_flag) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int row = index / cols;
    int col = index % cols;

    const scalar_t *offset_vertex1 = vertex1 + row * 2;
    const scalar_t *offset_vertex2 = vertex2 + col * 8;

    point point_[1];
    point polygon[4];

    point_[0].x = offset_vertex1[0];
    point_[0].y = offset_vertex1[1];

    polygon[0].x = offset_vertex2[0];
    polygon[0].y = offset_vertex2[1];
    polygon[1].x = offset_vertex2[2];
    polygon[1].y = offset_vertex2[3];
    polygon[2].x = offset_vertex2[4];
    polygon[2].y = offset_vertex2[5];
    polygon[3].x = offset_vertex2[6];
    polygon[3].y = offset_vertex2[7];

    int nCross = 0;
    int i, j;
    float sx, sy, tx, ty, px, py, x;
    for (i = 0, j = 3; i < 4; j = i, i++) {
      sx = polygon[i].x;
      sy = polygon[i].y;
      tx = polygon[j].x;
      ty = polygon[j].y;

      px = point_[0].x;
      py = point_[0].y;

      if (py < min(sy, ty)) continue;
      if (py > max(sy, ty)) continue;

      if ((sx == px && sy == py) || (tx == px && ty == py)) {
        break;
      } else {
        if ((sy < py && ty >= py) || (sy >= py && ty < py)) {
          x = sx + (py - sy) * (tx - sx) / (ty - sy);
          if (x == px) {
            break;
          }
          if (x > px) {
            nCross++;
          }
        }
      }
    }
    if (nCross % 2 == 1) {
      inside_flag[index] = 1.0;
    } else {
      inside_flag[index] = 0.0;
    }
    return;
  }
}

#endif  // POINTS_IN_POLYGONS_CUDA_KERNEL_CUH
