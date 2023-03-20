// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void min_area_polygons_impl(const Tensor pointsets, Tensor polygons) {
  DISPATCH_DEVICE_IMPL(min_area_polygons_impl, pointsets, polygons);
}

void min_area_polygons(const Tensor pointsets, Tensor polygons) {
  min_area_polygons_impl(pointsets, polygons);
}
