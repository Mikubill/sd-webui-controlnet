// Copyright (c) OpenMMLab. All rights reserved
#ifndef ACTIVE_ROTATED_FILTER_PYTORCH_H
#define ACTIVE_ROTATED_FILTER_PYTORCH_H
#include <torch/extension.h>
using namespace at;

void active_rotated_filter_forward(const Tensor input, const Tensor indices,
                                   Tensor output);

void active_rotated_filter_backward(const Tensor grad_out, const Tensor indices,
                                    Tensor grad_in);

#endif  // ACTIVE_ROTATED_FILTER_PYTORCH_H
