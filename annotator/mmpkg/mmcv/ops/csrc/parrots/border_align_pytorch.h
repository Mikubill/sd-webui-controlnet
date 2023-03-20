// Copyright (c) OpenMMLab. All rights reserved
#ifndef BORDER_ALIGN_PYTORCH_H
#define BORDER_ALIGN_PYTORCH_H
#include <torch/extension.h>
using namespace at;

#ifdef MMCV_WITH_CUDA
void border_align_forward_cuda(const Tensor &input, const Tensor &boxes,
                               Tensor output, Tensor argmax_idx,
                               const int pool_size);

void border_align_backward_cuda(const Tensor &grad_output, const Tensor &boxes,
                                const Tensor &argmax_idx, Tensor grad_input,
                                const int pool_size);
#endif

#endif  // BORDER_ALIGN_PYTORCH_H
