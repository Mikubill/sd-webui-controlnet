// Copyright (c) OpenMMLab. All rights reserved
#ifndef FOCAL_LOSS_PYTORCH_H
#define FOCAL_LOSS_PYTORCH_H
#include <torch/extension.h>
using namespace at;

void sigmoid_focal_loss_forward_cuda(Tensor input, Tensor target, Tensor weight,
                                     Tensor output, float gamma, float alpha);

void sigmoid_focal_loss_backward_cuda(Tensor input, Tensor target,
                                      Tensor weight, Tensor grad_input,
                                      float gamma, float alpha);

void softmax_focal_loss_forward_cuda(Tensor input, Tensor target, Tensor weight,
                                     Tensor output, float gamma, float alpha);

void softmax_focal_loss_backward_cuda(Tensor input, Tensor target,
                                      Tensor weight, Tensor buff,
                                      Tensor grad_input, float gamma,
                                      float alpha);
#endif  // FOCAL_LOSS_PYTORCH_H
