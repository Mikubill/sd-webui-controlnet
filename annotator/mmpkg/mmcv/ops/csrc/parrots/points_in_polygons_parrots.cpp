// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "points_in_polygons_pytorch.h"

using namespace parrots;

#ifdef MMCV_WITH_CUDA
void points_in_polygons_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                     const OperatorBase::in_list_t& ins,
                                     OperatorBase::out_list_t& outs) {
  auto points = buildATensor(ctx, ins[0]);
  auto polygons = buildATensor(ctx, ins[1]);

  auto output = buildATensor(ctx, outs[0]);

  points_in_polygons_forward(points, polygons, output);
}

PARROTS_EXTENSION_REGISTER(points_in_polygons_forward)
    .input(2)
    .output(1)
    .apply(points_in_polygons_cuda_parrots)
    .done();

#endif
