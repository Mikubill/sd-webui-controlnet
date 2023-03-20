// Copyright (c) OpenMMLab. All rights reserved
#ifndef BALL_QUERY_PYTORCH_H
#define BALL_QUERY_PYTORCH_H
#include <torch/extension.h>
using namespace at;

void ball_query_forward(const Tensor new_xyz, const Tensor xyz, Tensor idx,
                        int b, int n, int m, float min_radius, float max_radius,
                        int nsample);

#endif  // BALL_QUERY_PYTORCH_H
