// Copyright (c) OpenMMLab. All rights reserved
#ifndef IOU3D_CUDA_KERNEL_CUH
#define IOU3D_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

const int THREADS_PER_BLOCK_IOU3D = 16;
const int THREADS_PER_BLOCK_NMS = sizeof(unsigned long long) * 8;
__device__ const float EPS = 1e-8;

struct Point {
  float x, y;
  __device__ Point() {}
  __device__ Point(double _x, double _y) { x = _x, y = _y; }

  __device__ void set(float _x, float _y) {
    x = _x;
    y = _y;
  }

  __device__ Point operator+(const Point &b) const {
    return Point(x + b.x, y + b.y);
  }

  __device__ Point operator-(const Point &b) const {
    return Point(x - b.x, y - b.y);
  }
};

__device__ inline float cross(const Point &a, const Point &b) {
  return a.x * b.y - a.y * b.x;
}

__device__ inline float cross(const Point &p1, const Point &p2,
                              const Point &p0) {
  return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}

__device__ int check_rect_cross(const Point &p1, const Point &p2,
                                const Point &q1, const Point &q2) {
  int ret = min(p1.x, p2.x) <= max(q1.x, q2.x) &&
            min(q1.x, q2.x) <= max(p1.x, p2.x) &&
            min(p1.y, p2.y) <= max(q1.y, q2.y) &&
            min(q1.y, q2.y) <= max(p1.y, p2.y);
  return ret;
}

__device__ inline int check_in_box2d(const float *box, const Point &p) {
  // params: box (7) [x, y, z, dx, dy, dz, heading]
  const float MARGIN = 1e-2;

  float center_x = box[0], center_y = box[1];
  // rotate the point in the opposite direction of box
  float angle_cos = cos(-box[6]), angle_sin = sin(-box[6]);
  float rot_x = (p.x - center_x) * angle_cos + (p.y - center_y) * (-angle_sin);
  float rot_y = (p.x - center_x) * angle_sin + (p.y - center_y) * angle_cos;

  return (fabs(rot_x) < box[3] / 2 + MARGIN &&
          fabs(rot_y) < box[4] / 2 + MARGIN);
}

__device__ inline int intersection(const Point &p1, const Point &p0,
                                   const Point &q1, const Point &q0,
                                   Point &ans_point) {
  // fast exclusion
  if (check_rect_cross(p0, p1, q0, q1) == 0) return 0;

  // check cross standing
  float s1 = cross(q0, p1, p0);
  float s2 = cross(p1, q1, p0);
  float s3 = cross(p0, q1, q0);
  float s4 = cross(q1, p1, q0);

  if (!(s1 * s2 > 0 && s3 * s4 > 0)) return 0;

  // calculate intersection of two lines
  float s5 = cross(q1, p1, p0);
  if (fabs(s5 - s1) > EPS) {
    ans_point.x = (s5 * q0.x - s1 * q1.x) / (s5 - s1);
    ans_point.y = (s5 * q0.y - s1 * q1.y) / (s5 - s1);

  } else {
    float a0 = p0.y - p1.y, b0 = p1.x - p0.x, c0 = p0.x * p1.y - p1.x * p0.y;
    float a1 = q0.y - q1.y, b1 = q1.x - q0.x, c1 = q0.x * q1.y - q1.x * q0.y;
    float D = a0 * b1 - a1 * b0;

    ans_point.x = (b0 * c1 - b1 * c0) / D;
    ans_point.y = (a1 * c0 - a0 * c1) / D;
  }

  return 1;
}

__device__ inline void rotate_around_center(const Point &center,
                                            const float angle_cos,
                                            const float angle_sin, Point &p) {
  float new_x =
      (p.x - center.x) * angle_cos - (p.y - center.y) * angle_sin + center.x;
  float new_y =
      (p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y;
  p.set(new_x, new_y);
}

__device__ inline int point_cmp(const Point &a, const Point &b,
                                const Point &center) {
  return atan2(a.y - center.y, a.x - center.x) >
         atan2(b.y - center.y, b.x - center.x);
}

__device__ inline float box_overlap(const float *box_a, const float *box_b) {
  // params box_a: [x, y, z, dx, dy, dz, heading]
  // params box_b: [x, y, z, dx, dy, dz, heading]

  float a_angle = box_a[6], b_angle = box_b[6];
  float a_dx_half = box_a[3] / 2, b_dx_half = box_b[3] / 2,
        a_dy_half = box_a[4] / 2, b_dy_half = box_b[4] / 2;
  float a_x1 = box_a[0] - a_dx_half, a_y1 = box_a[1] - a_dy_half;
  float a_x2 = box_a[0] + a_dx_half, a_y2 = box_a[1] + a_dy_half;
  float b_x1 = box_b[0] - b_dx_half, b_y1 = box_b[1] - b_dy_half;
  float b_x2 = box_b[0] + b_dx_half, b_y2 = box_b[1] + b_dy_half;

  Point center_a(box_a[0], box_a[1]);
  Point center_b(box_b[0], box_b[1]);

  Point box_a_corners[5];
  box_a_corners[0].set(a_x1, a_y1);
  box_a_corners[1].set(a_x2, a_y1);
  box_a_corners[2].set(a_x2, a_y2);
  box_a_corners[3].set(a_x1, a_y2);

  Point box_b_corners[5];
  box_b_corners[0].set(b_x1, b_y1);
  box_b_corners[1].set(b_x2, b_y1);
  box_b_corners[2].set(b_x2, b_y2);
  box_b_corners[3].set(b_x1, b_y2);

  // get oriented corners
  float a_angle_cos = cos(a_angle), a_angle_sin = sin(a_angle);
  float b_angle_cos = cos(b_angle), b_angle_sin = sin(b_angle);

  for (int k = 0; k < 4; k++) {
    rotate_around_center(center_a, a_angle_cos, a_angle_sin, box_a_corners[k]);
    rotate_around_center(center_b, b_angle_cos, b_angle_sin, box_b_corners[k]);
  }

  box_a_corners[4] = box_a_corners[0];
  box_b_corners[4] = box_b_corners[0];

  // get intersection of lines
  Point cross_points[16];
  Point poly_center;
  int cnt = 0, flag = 0;

  poly_center.set(0, 0);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      flag = intersection(box_a_corners[i + 1], box_a_corners[i],
                          box_b_corners[j + 1], box_b_corners[j],
                          cross_points[cnt]);
      if (flag) {
        poly_center = poly_center + cross_points[cnt];
        cnt++;
      }
    }
  }

  // check corners
  for (int k = 0; k < 4; k++) {
    if (check_in_box2d(box_a, box_b_corners[k])) {
      poly_center = poly_center + box_b_corners[k];
      cross_points[cnt] = box_b_corners[k];
      cnt++;
    }
    if (check_in_box2d(box_b, box_a_corners[k])) {
      poly_center = poly_center + box_a_corners[k];
      cross_points[cnt] = box_a_corners[k];
      cnt++;
    }
  }

  poly_center.x /= cnt;
  poly_center.y /= cnt;

  // sort the points of polygon
  Point temp;
  for (int j = 0; j < cnt - 1; j++) {
    for (int i = 0; i < cnt - j - 1; i++) {
      if (point_cmp(cross_points[i], cross_points[i + 1], poly_center)) {
        temp = cross_points[i];
        cross_points[i] = cross_points[i + 1];
        cross_points[i + 1] = temp;
      }
    }
  }

  // get the overlap areas
  float area = 0;
  for (int k = 0; k < cnt - 1; k++) {
    area += cross(cross_points[k] - cross_points[0],
                  cross_points[k + 1] - cross_points[0]);
  }

  return fabs(area) / 2.0;
}

__device__ inline float iou_bev(const float *box_a, const float *box_b) {
  // params box_a: [x, y, z, dx, dy, dz, heading]
  // params box_b: [x, y, z, dx, dy, dz, heading]
  float sa = box_a[3] * box_a[4];
  float sb = box_b[3] * box_b[4];
  float s_overlap = box_overlap(box_a, box_b);
  return s_overlap / fmaxf(sa + sb - s_overlap, EPS);
}

__global__ void iou3d_boxes_overlap_bev_forward_cuda_kernel(
    const int num_a, const float *boxes_a, const int num_b,
    const float *boxes_b, float *ans_overlap) {
  // params boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
  // params boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]
  CUDA_2D_KERNEL_LOOP(b_idx, num_b, a_idx, num_a) {
    if (a_idx >= num_a || b_idx >= num_b) {
      return;
    }

    const float *cur_box_a = boxes_a + a_idx * 7;
    const float *cur_box_b = boxes_b + b_idx * 7;
    float cur_overlap = box_overlap(cur_box_a, cur_box_b);
    ans_overlap[a_idx * num_b + b_idx] = cur_overlap;
  }
}

__global__ void iou3d_nms3d_forward_cuda_kernel(const int boxes_num,
                                                const float nms_overlap_thresh,
                                                const float *boxes,
                                                unsigned long long *mask) {
  // params: boxes (N, 7) [x, y, z, dx, dy, dz, heading]
  // params: mask (N, N/THREADS_PER_BLOCK_NMS)
  const int blocks =
      (boxes_num + THREADS_PER_BLOCK_NMS - 1) / THREADS_PER_BLOCK_NMS;
  CUDA_2D_KERNEL_BLOCK_LOOP(col_start, blocks, row_start, blocks) {
    // if (row_start > col_start) return;

    const int row_size = fminf(boxes_num - row_start * THREADS_PER_BLOCK_NMS,
                               THREADS_PER_BLOCK_NMS);
    const int col_size = fminf(boxes_num - col_start * THREADS_PER_BLOCK_NMS,
                               THREADS_PER_BLOCK_NMS);

    __shared__ float block_boxes[THREADS_PER_BLOCK_NMS * 7];

    if (threadIdx.x < col_size) {
      block_boxes[threadIdx.x * 7 + 0] =
          boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 0];
      block_boxes[threadIdx.x * 7 + 1] =
          boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 1];
      block_boxes[threadIdx.x * 7 + 2] =
          boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 2];
      block_boxes[threadIdx.x * 7 + 3] =
          boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 3];
      block_boxes[threadIdx.x * 7 + 4] =
          boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 4];
      block_boxes[threadIdx.x * 7 + 5] =
          boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 5];
      block_boxes[threadIdx.x * 7 + 6] =
          boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 6];
    }
    __syncthreads();

    if (threadIdx.x < row_size) {
      const int cur_box_idx = THREADS_PER_BLOCK_NMS * row_start + threadIdx.x;
      const float *cur_box = boxes + cur_box_idx * 7;

      int i = 0;
      unsigned long long t = 0;
      int start = 0;
      if (row_start == col_start) {
        start = threadIdx.x + 1;
      }
      for (i = start; i < col_size; i++) {
        if (iou_bev(cur_box, block_boxes + i * 7) > nms_overlap_thresh) {
          t |= 1ULL << i;
        }
      }
      const int col_blocks =
          (boxes_num + THREADS_PER_BLOCK_NMS - 1) / THREADS_PER_BLOCK_NMS;
      mask[cur_box_idx * col_blocks + col_start] = t;
    }
  }
}

__device__ inline float iou_normal(float const *const a, float const *const b) {
  // params: a: [x, y, z, dx, dy, dz, heading]
  // params: b: [x, y, z, dx, dy, dz, heading]

  float left = fmaxf(a[0] - a[3] / 2, b[0] - b[3] / 2),
        right = fminf(a[0] + a[3] / 2, b[0] + b[3] / 2);
  float top = fmaxf(a[1] - a[4] / 2, b[1] - b[4] / 2),
        bottom = fminf(a[1] + a[4] / 2, b[1] + b[4] / 2);
  float width = fmaxf(right - left, 0.f), height = fmaxf(bottom - top, 0.f);
  float interS = width * height;
  float Sa = a[3] * a[4];
  float Sb = b[3] * b[4];
  return interS / fmaxf(Sa + Sb - interS, EPS);
}

__global__ void iou3d_nms3d_normal_forward_cuda_kernel(
    const int boxes_num, const float nms_overlap_thresh, const float *boxes,
    unsigned long long *mask) {
  // params: boxes (N, 7) [x, y, z, dx, dy, dz, heading]
  // params: mask (N, N/THREADS_PER_BLOCK_NMS)

  const int blocks =
      (boxes_num + THREADS_PER_BLOCK_NMS - 1) / THREADS_PER_BLOCK_NMS;
  CUDA_2D_KERNEL_BLOCK_LOOP(col_start, blocks, row_start, blocks) {
    // if (row_start > col_start) return;

    const int row_size = fminf(boxes_num - row_start * THREADS_PER_BLOCK_NMS,
                               THREADS_PER_BLOCK_NMS);
    const int col_size = fminf(boxes_num - col_start * THREADS_PER_BLOCK_NMS,
                               THREADS_PER_BLOCK_NMS);

    __shared__ float block_boxes[THREADS_PER_BLOCK_NMS * 7];

    if (threadIdx.x < col_size) {
      block_boxes[threadIdx.x * 7 + 0] =
          boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 0];
      block_boxes[threadIdx.x * 7 + 1] =
          boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 1];
      block_boxes[threadIdx.x * 7 + 2] =
          boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 2];
      block_boxes[threadIdx.x * 7 + 3] =
          boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 3];
      block_boxes[threadIdx.x * 7 + 4] =
          boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 4];
      block_boxes[threadIdx.x * 7 + 5] =
          boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 5];
      block_boxes[threadIdx.x * 7 + 6] =
          boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 7 + 6];
    }
    __syncthreads();

    if (threadIdx.x < row_size) {
      const int cur_box_idx = THREADS_PER_BLOCK_NMS * row_start + threadIdx.x;
      const float *cur_box = boxes + cur_box_idx * 7;

      int i = 0;
      unsigned long long t = 0;
      int start = 0;
      if (row_start == col_start) {
        start = threadIdx.x + 1;
      }
      for (i = start; i < col_size; i++) {
        if (iou_normal(cur_box, block_boxes + i * 7) > nms_overlap_thresh) {
          t |= 1ULL << i;
        }
      }
      const int col_blocks =
          (boxes_num + THREADS_PER_BLOCK_NMS - 1) / THREADS_PER_BLOCK_NMS;
      mask[cur_box_idx * col_blocks + col_start] = t;
    }
  }
}

#endif  // IOU3D_CUDA_KERNEL_CUH
