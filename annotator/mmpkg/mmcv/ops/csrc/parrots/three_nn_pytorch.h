// Copyright (c) OpenMMLab. All rights reserved
#ifndef THREE_NN_PYTORCH_H
#define THREE_NN_PYTORCH_H
#include <torch/extension.h>
using namespace at;

void three_nn_forward(Tensor unknown_tensor, Tensor known_tensor,
                      Tensor dist2_tensor, Tensor idx_tensor, int b, int n,
                      int m);
#endif  // THREE_NN_PYTORCH_H
