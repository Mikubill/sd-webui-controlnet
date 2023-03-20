// Copyright (c) OpenMMLab. All rights reserved
#ifndef PRROI_POOL_PYTORCH_H
#define PRROI_POOL_PYTORCH_H
#include <torch/extension.h>
using namespace at;

void prroi_pool_forward(Tensor input, Tensor rois, Tensor output,
                        int pooled_height, int pooled_width,
                        float spatial_scale);

void prroi_pool_backward(Tensor grad_output, Tensor rois, Tensor grad_input,
                         int pooled_height, int pooled_width,
                         float spatial_scale);

void prroi_pool_coor_backward(Tensor output, Tensor grad_output, Tensor input,
                              Tensor rois, Tensor grad_rois, int pooled_height,
                              int pooled_width, float spatial_scale);

#endif  // PRROI_POOL_PYTORCH_H
