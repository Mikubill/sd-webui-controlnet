// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void sync_bn_forward_mean_impl(const Tensor input, Tensor mean) {
  DISPATCH_DEVICE_IMPL(sync_bn_forward_mean_impl, input, mean);
}

void sync_bn_forward_var_impl(const Tensor input, const Tensor mean,
                              Tensor var) {
  DISPATCH_DEVICE_IMPL(sync_bn_forward_var_impl, input, mean, var);
}

void sync_bn_forward_output_impl(const Tensor input, const Tensor mean,
                                 const Tensor var, Tensor running_mean,
                                 Tensor running_var, const Tensor weight,
                                 const Tensor bias, Tensor norm, Tensor std,
                                 Tensor output, float eps, float momentum,
                                 int group_size) {
  DISPATCH_DEVICE_IMPL(sync_bn_forward_output_impl, input, mean, var,
                       running_mean, running_var, weight, bias, norm, std,
                       output, eps, momentum, group_size);
}

void sync_bn_backward_param_impl(const Tensor grad_output, const Tensor norm,
                                 Tensor grad_weight, Tensor grad_bias) {
  DISPATCH_DEVICE_IMPL(sync_bn_backward_param_impl, grad_output, norm,
                       grad_weight, grad_bias);
}

void sync_bn_backward_data_impl(const Tensor grad_output, const Tensor weight,
                                const Tensor grad_weight,
                                const Tensor grad_bias, const Tensor norm,
                                const Tensor std, Tensor grad_input) {
  DISPATCH_DEVICE_IMPL(sync_bn_backward_data_impl, grad_output, weight,
                       grad_weight, grad_bias, norm, std, grad_input);
}

void sync_bn_forward_mean(const Tensor input, Tensor mean) {
  sync_bn_forward_mean_impl(input, mean);
}

void sync_bn_forward_var(const Tensor input, const Tensor mean, Tensor var) {
  sync_bn_forward_var_impl(input, mean, var);
}

void sync_bn_forward_output(const Tensor input, const Tensor mean,
                            const Tensor var, const Tensor weight,
                            const Tensor bias, Tensor running_mean,
                            Tensor running_var, Tensor norm, Tensor std,
                            Tensor output, float eps, float momentum,
                            int group_size) {
  sync_bn_forward_output_impl(input, mean, var, running_mean, running_var,
                              weight, bias, norm, std, output, eps, momentum,
                              group_size);
}

void sync_bn_backward_param(const Tensor grad_output, const Tensor norm,
                            Tensor grad_weight, Tensor grad_bias) {
  sync_bn_backward_param_impl(grad_output, norm, grad_weight, grad_bias);
}

void sync_bn_backward_data(const Tensor grad_output, const Tensor weight,
                           const Tensor grad_weight, const Tensor grad_bias,
                           const Tensor norm, const Tensor std,
                           Tensor grad_input) {
  sync_bn_backward_data_impl(grad_output, weight, grad_weight, grad_bias, norm,
                             std, grad_input);
}
