// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "min_area_polygons_pytorch.h"

using namespace parrots;

#ifdef MMCV_WITH_CUDA
void min_area_polygons_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                    const OperatorBase::in_list_t& ins,
                                    OperatorBase::out_list_t& outs) {
  auto pointsets = buildATensor(ctx, ins[0]);

  auto polygons = buildATensor(ctx, outs[0]);
  min_area_polygons(pointsets, polygons);
}

PARROTS_EXTENSION_REGISTER(min_area_polygons)
    .input(1)
    .output(1)
    .apply(min_area_polygons_cuda_parrots)
    .done();

#endif
