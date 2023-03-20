// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void sigmoid_focal_loss_forward_impl(Tensor input, Tensor target, Tensor weight,
                                     Tensor output, float gamma, float alpha) {
  DISPATCH_DEVICE_IMPL(sigmoid_focal_loss_forward_impl, input, target, weight,
                       output, gamma, alpha);
}

void sigmoid_focal_loss_backward_impl(Tensor input, Tensor target,
                                      Tensor weight, Tensor grad_input,
                                      float gamma, float alpha) {
  DISPATCH_DEVICE_IMPL(sigmoid_focal_loss_backward_impl, input, target, weight,
                       grad_input, gamma, alpha);
}

void softmax_focal_loss_forward_impl(Tensor input, Tensor target, Tensor weight,
                                     Tensor output, float gamma, float alpha) {
  DISPATCH_DEVICE_IMPL(softmax_focal_loss_forward_impl, input, target, weight,
                       output, gamma, alpha);
}

void softmax_focal_loss_backward_impl(Tensor input, Tensor target,
                                      Tensor weight, Tensor buff,
                                      Tensor grad_input, float gamma,
                                      float alpha) {
  DISPATCH_DEVICE_IMPL(softmax_focal_loss_backward_impl, input, target, weight,
                       buff, grad_input, gamma, alpha);
}

void sigmoid_focal_loss_forward(Tensor input, Tensor target, Tensor weight,
                                Tensor output, float gamma, float alpha) {
  sigmoid_focal_loss_forward_impl(input, target, weight, output, gamma, alpha);
}

void sigmoid_focal_loss_backward(Tensor input, Tensor target, Tensor weight,
                                 Tensor grad_input, float gamma, float alpha) {
  sigmoid_focal_loss_backward_impl(input, target, weight, grad_input, gamma,
                                   alpha);
}

void softmax_focal_loss_forward(Tensor input, Tensor target, Tensor weight,
                                Tensor output, float gamma, float alpha) {
  softmax_focal_loss_forward_impl(input, target, weight, output, gamma, alpha);
}

void softmax_focal_loss_backward(Tensor input, Tensor target, Tensor weight,
                                 Tensor buff, Tensor grad_input, float gamma,
                                 float alpha) {
  softmax_focal_loss_backward_impl(input, target, weight, buff, grad_input,
                                   gamma, alpha);
}
