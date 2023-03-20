// Copyright (c) OpenMMLab. All rights reserved
#ifndef ROI_POOL_PYTORCH_H
#define ROI_POOL_PYTORCH_H
#include <torch/extension.h>
using namespace at;

#ifdef MMCV_WITH_CUDA
void roi_pool_forward_cuda(Tensor input, Tensor rois, Tensor output,
                           Tensor argmax, int pooled_height, int pooled_width,
                           float spatial_scale);

void roi_pool_backward_cuda(Tensor grad_output, Tensor rois, Tensor argmax,
                            Tensor grad_input, int pooled_height,
                            int pooled_width, float spatial_scale);
#endif
#endif  // ROI_POOL_PYTORCH_H
