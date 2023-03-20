// Copyright 2019 Yan Yan
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <math.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <iostream>

namespace py = pybind11;
using namespace pybind11::literals;

template <typename DType, int NDim>
int points_to_voxel_3d_np(py::array_t<DType> points, py::array_t<DType> voxels,
                          py::array_t<int> coors,
                          py::array_t<int> num_points_per_voxel,
                          py::array_t<int> coor_to_voxelidx,
                          std::vector<DType> voxel_size,
                          std::vector<DType> coors_range, int max_points,
                          int max_voxels) {
  auto points_rw = points.template mutable_unchecked<2>();
  auto voxels_rw = voxels.template mutable_unchecked<3>();
  auto coors_rw = coors.mutable_unchecked<2>();
  auto num_points_per_voxel_rw = num_points_per_voxel.mutable_unchecked<1>();
  auto coor_to_voxelidx_rw = coor_to_voxelidx.mutable_unchecked<NDim>();
  auto N = points_rw.shape(0);
  auto num_features = points_rw.shape(1);
  constexpr int ndim_minus_1 = NDim - 1;
  int voxel_num = 0;
  bool failed = false;
  int coor[NDim];
  int c;
  int grid_size[NDim];
  for (int i = 0; i < NDim; ++i) {
    grid_size[i] =
        round((coors_range[NDim + i] - coors_range[i]) / voxel_size[i]);
  }
  int voxelidx, num;
  for (int i = 0; i < N; ++i) {
    failed = false;
    for (int j = 0; j < NDim; ++j) {
      c = floor((points_rw(i, j) - coors_range[j]) / voxel_size[j]);
      if ((c < 0 || c >= grid_size[j])) {
        failed = true;
        break;
      }
      coor[ndim_minus_1 - j] = c;
    }
    if (failed) continue;
    voxelidx = coor_to_voxelidx_rw(coor[0], coor[1], coor[2]);
    if (voxelidx == -1) {
      voxelidx = voxel_num;
      if (voxel_num >= max_voxels) continue;
      voxel_num += 1;
      coor_to_voxelidx_rw(coor[0], coor[1], coor[2]) = voxelidx;
      for (int k = 0; k < NDim; ++k) {
        coors_rw(voxelidx, k) = coor[k];
      }
    }
    num = num_points_per_voxel_rw(voxelidx);
    if (num < max_points) {
      for (int k = 0; k < num_features; ++k) {
        voxels_rw(voxelidx, num, k) = points_rw(i, k);
      }
      num_points_per_voxel_rw(voxelidx) += 1;
    }
  }
  for (int i = 0; i < voxel_num; ++i) {
    coor_to_voxelidx_rw(coors_rw(i, 0), coors_rw(i, 1), coors_rw(i, 2)) = -1;
  }
  return voxel_num;
}

template <typename DType, int NDim>
int points_to_voxel_3d_np_mean(py::array_t<DType> points,
                               py::array_t<DType> voxels,
                               py::array_t<DType> means, py::array_t<int> coors,
                               py::array_t<int> num_points_per_voxel,
                               py::array_t<int> coor_to_voxelidx,
                               std::vector<DType> voxel_size,
                               std::vector<DType> coors_range, int max_points,
                               int max_voxels) {
  auto points_rw = points.template mutable_unchecked<2>();
  auto means_rw = means.template mutable_unchecked<2>();
  auto voxels_rw = voxels.template mutable_unchecked<3>();
  auto coors_rw = coors.mutable_unchecked<2>();
  auto num_points_per_voxel_rw = num_points_per_voxel.mutable_unchecked<1>();
  auto coor_to_voxelidx_rw = coor_to_voxelidx.mutable_unchecked<NDim>();
  auto N = points_rw.shape(0);
  auto num_features = points_rw.shape(1);
  constexpr int ndim_minus_1 = NDim - 1;
  int voxel_num = 0;
  bool failed = false;
  int coor[NDim];
  int c;
  int grid_size[NDim];
  for (int i = 0; i < NDim; ++i) {
    grid_size[i] =
        round((coors_range[NDim + i] - coors_range[i]) / voxel_size[i]);
  }
  int voxelidx, num;
  for (int i = 0; i < N; ++i) {
    failed = false;
    for (int j = 0; j < NDim; ++j) {
      c = floor((points_rw(i, j) - coors_range[j]) / voxel_size[j]);
      if ((c < 0 || c >= grid_size[j])) {
        failed = true;
        break;
      }
      coor[ndim_minus_1 - j] = c;
    }
    if (failed) continue;
    voxelidx = coor_to_voxelidx_rw(coor[0], coor[1], coor[2]);
    if (voxelidx == -1) {
      voxelidx = voxel_num;
      if (voxel_num >= max_voxels) continue;
      voxel_num += 1;
      coor_to_voxelidx_rw(coor[0], coor[1], coor[2]) = voxelidx;
      for (int k = 0; k < NDim; ++k) {
        coors_rw(voxelidx, k) = coor[k];
      }
    }
    num = num_points_per_voxel_rw(voxelidx);
    if (num < max_points) {
      for (int k = 0; k < num_features; ++k) {
        voxels_rw(voxelidx, num, k) = points_rw(i, k);
      }
      num_points_per_voxel_rw(voxelidx) += 1;
      for (int k = 0; k < num_features; ++k) {
        means_rw(voxelidx, k) +=
            (points_rw(i, k) - means_rw(voxelidx, k)) / DType(num + 1);
      }
    }
  }
  for (int i = 0; i < voxel_num; ++i) {
    coor_to_voxelidx_rw(coors_rw(i, 0), coors_rw(i, 1), coors_rw(i, 2)) = -1;
    num = num_points_per_voxel_rw(i);
    for (int j = num; j < max_points; ++j) {
      for (int k = 0; k < num_features; ++k) {
        voxels_rw(i, j, k) = means_rw(i, k);
      }
    }
  }
  return voxel_num;
}

template <typename DType, int NDim>
int points_to_voxel_3d_np_height(
    py::array_t<DType> points, py::array_t<DType> voxels,
    py::array_t<DType> height, py::array_t<DType> maxs, py::array_t<int> coors,
    py::array_t<int> num_points_per_voxel, py::array_t<int> coor_to_voxelidx,
    std::vector<DType> voxel_size, std::vector<DType> coors_range,
    int max_points, int max_voxels) {
  auto points_rw = points.template mutable_unchecked<2>();
  auto height_rw = height.template mutable_unchecked<2>();
  auto maxs_rw = maxs.template mutable_unchecked<2>();
  auto voxels_rw = voxels.template mutable_unchecked<3>();
  auto coors_rw = coors.mutable_unchecked<2>();
  auto num_points_per_voxel_rw = num_points_per_voxel.mutable_unchecked<1>();
  auto coor_to_voxelidx_rw = coor_to_voxelidx.mutable_unchecked<NDim>();
  auto N = points_rw.shape(0);
  auto num_features = points_rw.shape(1);
  constexpr int ndim_minus_1 = NDim - 1;
  int voxel_num = 0;
  bool failed = false;
  int coor[NDim];
  int c;
  int grid_size[NDim];
  for (int i = 0; i < NDim; ++i) {
    grid_size[i] =
        round((coors_range[NDim + i] - coors_range[i]) / voxel_size[i]);
  }
  int voxelidx, num;
  for (int i = 0; i < N; ++i) {
    failed = false;
    for (int j = 0; j < NDim; ++j) {
      c = floor((points_rw(i, j) - coors_range[j]) / voxel_size[j]);
      if ((c < 0 || c >= grid_size[j])) {
        failed = true;
        break;
      }
      coor[ndim_minus_1 - j] = c;
    }
    if (failed) continue;
    voxelidx = coor_to_voxelidx_rw(coor[0], coor[1], coor[2]);
    if (voxelidx == -1) {
      voxelidx = voxel_num;
      if (voxel_num >= max_voxels) continue;
      voxel_num += 1;
      coor_to_voxelidx_rw(coor[0], coor[1], coor[2]) = voxelidx;
      for (int k = 0; k < NDim; ++k) {
        coors_rw(voxelidx, k) = coor[k];
      }
    }
    num = num_points_per_voxel_rw(voxelidx);
    if (num < max_points) {
      for (int k = 0; k < num_features; ++k) {
        voxels_rw(voxelidx, num, k) = points_rw(i, k);
        height_rw(voxelidx, k) =
            std::min(points_rw(i, k), height_rw(voxelidx, k));
        maxs_rw(voxelidx, k) = std::max(points_rw(i, k), maxs_rw(voxelidx, k));
      }
      num_points_per_voxel_rw(voxelidx) += 1;
    }
  }
  for (int i = 0; i < voxel_num; ++i) {
    coor_to_voxelidx_rw(coors_rw(i, 0), coors_rw(i, 1), coors_rw(i, 2)) = -1;
    for (int k = 0; k < num_features; ++k) {
      height_rw(i, k) = maxs_rw(i, k) - height_rw(i, k);
    }
  }
  return voxel_num;
}

template <typename DType, int NDim>
int block_filtering(py::array_t<DType> points, py::array_t<int> mask,
                    py::array_t<DType> height, py::array_t<DType> maxs,
                    py::array_t<int> coor_to_voxelidx,
                    std::vector<DType> voxel_size,
                    std::vector<DType> coors_range, int max_voxels, DType eps) {
  auto points_rw = points.template mutable_unchecked<2>();
  auto height_rw = height.template mutable_unchecked<1>();
  auto maxs_rw = maxs.template mutable_unchecked<1>();
  auto coor_to_voxelidx_rw = coor_to_voxelidx.mutable_unchecked<NDim>();
  auto N = points_rw.shape(0);
  auto num_features = points_rw.shape(1);
  constexpr int ndim_minus_1 = NDim - 1;
  int voxel_num = 0;
  bool failed = false;
  int coor[NDim];
  int c;
  int grid_size[NDim];
  for (int i = 0; i < NDim; ++i) {
    grid_size[i] =
        round((coors_range[NDim + i] - coors_range[i]) / voxel_size[i]);
  }
  int voxelidx, num;
  for (int i = 0; i < N; ++i) {
    failed = false;
    for (int j = 0; j < NDim; ++j) {
      c = floor((points_rw(i, j) - coors_range[j]) / voxel_size[j]);
      if ((c < 0 || c >= grid_size[j])) {
        failed = true;
        break;
      }
      coor[ndim_minus_1 - j] = c;
    }
    if (failed) continue;
    voxelidx = coor_to_voxelidx_rw(coor[0], coor[1], coor[2]);
    if (voxelidx == -1) {
      voxelidx = voxel_num;
      voxel_num += 1;
      coor_to_voxelidx_rw(coor[0], coor[1], coor[2]) = voxelidx;
    }
    height_rw(voxelidx) = std::min(points_rw(i, 2), height_rw(voxelidx));
    maxs_rw(voxelidx) = std::max(points_rw(i, 2), maxs_rw(voxelidx));
  }
  for (int i = 0; i < N; ++i) {
    failed = false;
    for (int j = 0; j < NDim; ++j) {
      c = floor((points_rw(i, j) - coors_range[j]) / voxel_size[j]);
      if ((c < 0 || c >= grid_size[j])) {
        failed = true;
        break;
      }
      coor[ndim_minus_1 - j] = c;
    }
    if (failed) continue;
    voxelidx = coor_to_voxelidx_rw(coor[0], coor[1], coor[2]);
    if ((maxs_rw(voxelidx) - height_rw(voxelidx, 2)) < eps) {
      mask(i) = 0;
    }
  }
}

template <typename DType, int NDim>
int points_to_voxel_3d_with_filtering(
    py::array_t<DType> points, py::array_t<DType> voxels,
    py::array_t<int> voxel_mask, py::array_t<DType> mins,
    py::array_t<DType> maxs, py::array_t<int> coors,
    py::array_t<int> num_points_per_voxel, py::array_t<int> coor_to_voxelidx,
    std::vector<DType> voxel_size, std::vector<DType> coors_range,
    int max_points, int max_voxels, int block_factor, int block_size,
    DType height_threshold) {
  auto points_rw = points.template mutable_unchecked<2>();
  auto mins_rw = mins.template mutable_unchecked<2>();
  auto maxs_rw = maxs.template mutable_unchecked<2>();
  auto voxels_rw = voxels.template mutable_unchecked<3>();
  auto voxel_mask_rw = voxel_mask.template mutable_unchecked<1>();
  auto coors_rw = coors.mutable_unchecked<2>();
  auto num_points_per_voxel_rw = num_points_per_voxel.mutable_unchecked<1>();
  auto coor_to_voxelidx_rw = coor_to_voxelidx.mutable_unchecked<NDim>();
  auto N = points_rw.shape(0);
  auto num_features = points_rw.shape(1);
  constexpr int ndim_minus_1 = NDim - 1;
  int voxel_num = 0;
  bool failed = false;
  int coor[NDim];
  int c;
  int grid_size[NDim];

  DType max_value, min_value;
  for (int i = 0; i < NDim; ++i) {
    grid_size[i] =
        round((coors_range[NDim + i] - coors_range[i]) / voxel_size[i]);
  }
  int block_shape_H = grid_size[1] / block_factor;
  int block_shape_W = grid_size[0] / block_factor;
  int voxelidx, num;
  int block_coor[2];
  int startx, stopx, starty, stopy;
  for (int i = 0; i < N; ++i) {
    failed = false;
    for (int j = 0; j < NDim; ++j) {
      c = floor((points_rw(i, j) - coors_range[j]) / voxel_size[j]);
      if ((c < 0 || c >= grid_size[j])) {
        failed = true;
        break;
      }
      coor[ndim_minus_1 - j] = c;
    }
    if (failed) continue;
    voxelidx = coor_to_voxelidx_rw(coor[0], coor[1], coor[2]);
    if (voxelidx == -1) {
      voxelidx = voxel_num;
      if (voxel_num >= max_voxels) continue;
      voxel_num += 1;
      coor_to_voxelidx_rw(coor[0], coor[1], coor[2]) = voxelidx;
      for (int k = 0; k < NDim; ++k) {
        coors_rw(voxelidx, k) = coor[k];
      }
    }
    num = num_points_per_voxel_rw(voxelidx);
    if (num < max_points) {
      for (int k = 0; k < num_features; ++k) {
        voxels_rw(voxelidx, num, k) = points_rw(i, k);
      }
      block_coor[0] = coor[1] / block_factor;
      block_coor[1] = coor[2] / block_factor;
      mins_rw(block_coor[0], block_coor[1]) =
          std::min(points_rw(i, 2), mins_rw(block_coor[0], block_coor[1]));
      maxs_rw(block_coor[0], block_coor[1]) =
          std::max(points_rw(i, 2), maxs_rw(block_coor[0], block_coor[1]));
      num_points_per_voxel_rw(voxelidx) += 1;
    }
  }
  for (int i = 0; i < voxel_num; ++i) {
    coor[1] = coors_rw(i, 1);
    coor[2] = coors_rw(i, 2);
    coor_to_voxelidx_rw(coors_rw(i, 0), coor[1], coor[2]) = -1;
    block_coor[0] = coor[1] / block_factor;
    block_coor[1] = coor[2] / block_factor;
    min_value = mins_rw(block_coor[0], block_coor[1]);
    max_value = maxs_rw(block_coor[0], block_coor[1]);
    startx = std::max(0, block_coor[0] - block_size / 2);
    stopx =
        std::min(block_shape_H, block_coor[0] + block_size - block_size / 2);
    starty = std::max(0, block_coor[1] - block_size / 2);
    stopy =
        std::min(block_shape_W, block_coor[1] + block_size - block_size / 2);

    for (int j = startx; j < stopx; ++j) {
      for (int k = starty; k < stopy; ++k) {
        min_value = std::min(min_value, mins_rw(j, k));
        max_value = std::max(max_value, maxs_rw(j, k));
      }
    }
    voxel_mask_rw(i) = (max_value - min_value) > height_threshold;
  }
  return voxel_num;
}
