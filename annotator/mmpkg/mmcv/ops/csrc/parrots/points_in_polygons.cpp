#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void points_in_polygons_forward_impl(const Tensor points, const Tensor polygons,
                                     Tensor output, const int rows,
                                     const int cols) {
  DISPATCH_DEVICE_IMPL(points_in_polygons_forward_impl, points, polygons,
                       output, rows, cols);
}

void points_in_polygons_forward(Tensor points, Tensor polygons, Tensor output) {
  int rows = points.size(0);
  int cols = polygons.size(0);
  points_in_polygons_forward_impl(points, polygons, output, rows, cols);
}
