// Copyright (c) OpenMMLab. All rights reserved
// It is modified from https://github.com/whai362/PSENet
#include <iostream>
#include <queue>

#include "pytorch_cpp_helper.hpp"

using namespace std;

class Point2d {
 public:
  int x;
  int y;

  Point2d() : x(0), y(0) {}
  Point2d(int _x, int _y) : x(_x), y(_y) {}
};

void kernel_dilate(const uint8_t *data, IntArrayRef data_shape,
                   const int *label_map, int &label_num, int &min_area,
                   vector<vector<int>> &text_line) {
  std::vector<int> area(label_num + 1);
  int kernel_num = data_shape[0];
  int height = data_shape[1];
  int width = data_shape[2];

  for (int x = 0; x < height; ++x) {
    for (int y = 0; y < width; ++y) {
      int label = label_map[x * width + y];
      if (label == 0) continue;
      area[label] += 1;
    }
  }

  queue<Point2d> queue, next_queue;
  for (int x = 0; x < height; ++x) {
    vector<int> row(width);
    for (int y = 0; y < width; ++y) {
      int label = label_map[x * width + y];
      if (label == 0) continue;
      if (area[label] < min_area) continue;

      Point2d point(x, y);
      queue.push(point);
      row[y] = label;
    }
    text_line.emplace_back(row);
  }

  int dx[] = {-1, 1, 0, 0};
  int dy[] = {0, 0, -1, 1};
  vector<int> kernel_step(kernel_num);
  std::for_each(kernel_step.begin(), kernel_step.end(),
                [=](int &k) { return k * height * width; });

  for (int kernel_id = kernel_num - 2; kernel_id >= 0; --kernel_id) {
    while (!queue.empty()) {
      Point2d point = queue.front();
      queue.pop();
      int x = point.x;
      int y = point.y;
      int label = text_line[x][y];

      bool is_edge = true;
      for (int d = 0; d < 4; ++d) {
        int tmp_x = x + dx[d];
        int tmp_y = y + dy[d];

        if (tmp_x < 0 || tmp_x >= height) continue;
        if (tmp_y < 0 || tmp_y >= width) continue;
        int kernel_value = data[kernel_step[kernel_id] + tmp_x * width + tmp_y];
        if (kernel_value == 0) continue;
        if (text_line[tmp_x][tmp_y] > 0) continue;

        Point2d point(tmp_x, tmp_y);
        queue.push(point);
        text_line[tmp_x][tmp_y] = label;
        is_edge = false;
      }

      if (is_edge) {
        next_queue.push(point);
      }
    }
    swap(queue, next_queue);
  }
}

std::vector<std::vector<int>> contour_expand(Tensor kernel_mask,
                                             Tensor internal_kernel_label,
                                             int min_kernel_area,
                                             int kernel_num) {
  kernel_mask = kernel_mask.contiguous();
  internal_kernel_label = internal_kernel_label.contiguous();
  assert(kernel_mask.dim() == 3);
  assert(internal_kernel_label.dim() == 2);
  assert(kernel_mask.size(1) == internal_kernel_label.size(0));
  assert(kernel_mask.size(2) == internal_kernel_label.size(1));
  CHECK_CPU_INPUT(kernel_mask);
  CHECK_CPU_INPUT(internal_kernel_label);
  auto ptr_data = kernel_mask.data_ptr<uint8_t>();
  IntArrayRef data_shape = kernel_mask.sizes();

  auto data_label_map = internal_kernel_label.data_ptr<int32_t>();
  vector<vector<int>> text_line;

  kernel_dilate(ptr_data, data_shape, data_label_map, kernel_num,
                min_kernel_area, text_line);

  return text_line;
}
