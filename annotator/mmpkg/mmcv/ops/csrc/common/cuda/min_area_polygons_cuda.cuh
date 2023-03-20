// Copyright (c) OpenMMLab. All rights reserved
#ifndef MIN_AREA_POLYGONS_CUDA_KERNEL_CUH
#define MIN_AREA_POLYGONS_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

#define MAXN 20
__device__ const float PI = 3.1415926;

struct Point {
  float x, y;
  __device__ Point() {}
  __device__ Point(float x, float y) : x(x), y(y) {}
};

__device__ inline void swap1(Point *a, Point *b) {
  Point temp;
  temp.x = a->x;
  temp.y = a->y;

  a->x = b->x;
  a->y = b->y;

  b->x = temp.x;
  b->y = temp.y;
}
__device__ inline float cross(Point o, Point a, Point b) {
  return (a.x - o.x) * (b.y - o.y) - (b.x - o.x) * (a.y - o.y);
}

__device__ inline float dis(Point a, Point b) {
  return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
}
__device__ inline void minBoundingRect(Point *ps, int n_points, float *minbox) {
  float convex_points[2][MAXN];
  for (int j = 0; j < n_points; j++) {
    convex_points[0][j] = ps[j].x;
  }
  for (int j = 0; j < n_points; j++) {
    convex_points[1][j] = ps[j].y;
  }

  Point edges[MAXN];
  float edges_angles[MAXN];
  float unique_angles[MAXN];
  int n_edges = n_points - 1;
  int n_unique = 0;
  int unique_flag = 0;

  for (int i = 0; i < n_edges; i++) {
    edges[i].x = ps[i + 1].x - ps[i].x;
    edges[i].y = ps[i + 1].y - ps[i].y;
  }
  for (int i = 0; i < n_edges; i++) {
    edges_angles[i] = atan2((double)edges[i].y, (double)edges[i].x);
    if (edges_angles[i] >= 0) {
      edges_angles[i] = fmod((double)edges_angles[i], (double)PI / 2);
    } else {
      edges_angles[i] =
          edges_angles[i] - (int)(edges_angles[i] / (PI / 2) - 1) * (PI / 2);
    }
  }
  unique_angles[0] = edges_angles[0];
  n_unique += 1;
  for (int i = 1; i < n_edges; i++) {
    for (int j = 0; j < n_unique; j++) {
      if (edges_angles[i] == unique_angles[j]) {
        unique_flag += 1;
      }
    }
    if (unique_flag == 0) {
      unique_angles[n_unique] = edges_angles[i];
      n_unique += 1;
      unique_flag = 0;
    } else {
      unique_flag = 0;
    }
  }

  float minarea = 1e12;
  for (int i = 0; i < n_unique; i++) {
    float R[2][2];
    float rot_points[2][MAXN];
    R[0][0] = cos(unique_angles[i]);
    R[0][1] = sin(unique_angles[i]);
    R[1][0] = -sin(unique_angles[i]);
    R[1][1] = cos(unique_angles[i]);
    // R x Points
    for (int m = 0; m < 2; m++) {
      for (int n = 0; n < n_points; n++) {
        float sum = 0.0;
        for (int k = 0; k < 2; k++) {
          sum = sum + R[m][k] * convex_points[k][n];
        }
        rot_points[m][n] = sum;
      }
    }

    // xmin;
    float xmin, ymin, xmax, ymax;
    xmin = 1e12;
    for (int j = 0; j < n_points; j++) {
      if (isinf(rot_points[0][j]) || isnan(rot_points[0][j])) {
        continue;
      } else {
        if (rot_points[0][j] < xmin) {
          xmin = rot_points[0][j];
        }
      }
    }
    // ymin
    ymin = 1e12;
    for (int j = 0; j < n_points; j++) {
      if (isinf(rot_points[1][j]) || isnan(rot_points[1][j])) {
        continue;
      } else {
        if (rot_points[1][j] < ymin) {
          ymin = rot_points[1][j];
        }
      }
    }
    // xmax
    xmax = -1e12;
    for (int j = 0; j < n_points; j++) {
      if (isinf(rot_points[0][j]) || isnan(rot_points[0][j])) {
        continue;
      } else {
        if (rot_points[0][j] > xmax) {
          xmax = rot_points[0][j];
        }
      }
    }
    // ymax
    ymax = -1e12;
    for (int j = 0; j < n_points; j++) {
      if (isinf(rot_points[1][j]) || isnan(rot_points[1][j])) {
        continue;
      } else {
        if (rot_points[1][j] > ymax) {
          ymax = rot_points[1][j];
        }
      }
    }
    float area = (xmax - xmin) * (ymax - ymin);
    if (area < minarea) {
      minarea = area;
      minbox[0] = unique_angles[i];
      minbox[1] = xmin;
      minbox[2] = ymin;
      minbox[3] = xmax;
      minbox[4] = ymax;
    }
  }
}

// convex_find
__device__ inline void Jarvis(Point *in_poly, int &n_poly) {
  int n_input = n_poly;
  Point input_poly[20];
  for (int i = 0; i < n_input; i++) {
    input_poly[i].x = in_poly[i].x;
    input_poly[i].y = in_poly[i].y;
  }
  Point p_max, p_k;
  int max_index, k_index;
  int Stack[20], top1, top2;
  // float sign;
  double sign;
  Point right_point[10], left_point[10];

  for (int i = 0; i < n_poly; i++) {
    if (in_poly[i].y < in_poly[0].y ||
        in_poly[i].y == in_poly[0].y && in_poly[i].x < in_poly[0].x) {
      Point *j = &(in_poly[0]);
      Point *k = &(in_poly[i]);
      swap1(j, k);
    }
    if (i == 0) {
      p_max = in_poly[0];
      max_index = 0;
    }
    if (in_poly[i].y > p_max.y ||
        in_poly[i].y == p_max.y && in_poly[i].x > p_max.x) {
      p_max = in_poly[i];
      max_index = i;
    }
  }
  if (max_index == 0) {
    max_index = 1;
    p_max = in_poly[max_index];
  }

  k_index = 0, Stack[0] = 0, top1 = 0;
  while (k_index != max_index) {
    p_k = p_max;
    k_index = max_index;
    for (int i = 1; i < n_poly; i++) {
      sign = cross(in_poly[Stack[top1]], in_poly[i], p_k);
      if ((sign > 0) || ((sign == 0) && (dis(in_poly[Stack[top1]], in_poly[i]) >
                                         dis(in_poly[Stack[top1]], p_k)))) {
        p_k = in_poly[i];
        k_index = i;
      }
    }
    top1++;
    Stack[top1] = k_index;
  }

  for (int i = 0; i <= top1; i++) {
    right_point[i] = in_poly[Stack[i]];
  }

  k_index = 0, Stack[0] = 0, top2 = 0;

  while (k_index != max_index) {
    p_k = p_max;
    k_index = max_index;
    for (int i = 1; i < n_poly; i++) {
      sign = cross(in_poly[Stack[top2]], in_poly[i], p_k);
      if ((sign < 0) || (sign == 0) && (dis(in_poly[Stack[top2]], in_poly[i]) >
                                        dis(in_poly[Stack[top2]], p_k))) {
        p_k = in_poly[i];
        k_index = i;
      }
    }
    top2++;
    Stack[top2] = k_index;
  }

  for (int i = top2 - 1; i >= 0; i--) {
    left_point[i] = in_poly[Stack[i]];
  }

  for (int i = 0; i < top1 + top2; i++) {
    if (i <= top1) {
      in_poly[i] = right_point[i];
    } else {
      in_poly[i] = left_point[top2 - (i - top1)];
    }
  }
  n_poly = top1 + top2;
}

template <typename T>
__device__ inline void Findminbox(T const *const p, T *minpoints) {
  Point ps1[MAXN];
  Point convex[MAXN];
  for (int i = 0; i < 9; i++) {
    convex[i].x = p[i * 2];
    convex[i].y = p[i * 2 + 1];
  }
  int n_convex = 9;
  Jarvis(convex, n_convex);
  int n1 = n_convex;
  for (int i = 0; i < n1; i++) {
    ps1[i].x = convex[i].x;
    ps1[i].y = convex[i].y;
  }
  ps1[n1].x = convex[0].x;
  ps1[n1].y = convex[0].y;

  float minbbox[5] = {0};
  minBoundingRect(ps1, n1 + 1, minbbox);
  float angle = minbbox[0];
  float xmin = minbbox[1];
  float ymin = minbbox[2];
  float xmax = minbbox[3];
  float ymax = minbbox[4];
  float R[2][2];

  R[0][0] = cos(angle);
  R[0][1] = sin(angle);
  R[1][0] = -sin(angle);
  R[1][1] = cos(angle);

  minpoints[0] = xmax * R[0][0] + ymin * R[1][0];
  minpoints[1] = xmax * R[0][1] + ymin * R[1][1];
  minpoints[2] = xmin * R[0][0] + ymin * R[1][0];
  minpoints[3] = xmin * R[0][1] + ymin * R[1][1];
  minpoints[4] = xmin * R[0][0] + ymax * R[1][0];
  minpoints[5] = xmin * R[0][1] + ymax * R[1][1];
  minpoints[6] = xmax * R[0][0] + ymax * R[1][0];
  minpoints[7] = xmax * R[0][1] + ymax * R[1][1];
}

template <typename T>
__global__ void min_area_polygons_cuda_kernel(const int ex_n_boxes,
                                              const T *ex_boxes, T *minbox) {
  CUDA_1D_KERNEL_LOOP(index, ex_n_boxes) {
    const T *cur_box = ex_boxes + index * 18;
    T *cur_min_box = minbox + index * 8;
    Findminbox(cur_box, cur_min_box);
  }
}

#endif  // MIN_AREA_POLYGONS_CUDA_KERNEL_CUH
