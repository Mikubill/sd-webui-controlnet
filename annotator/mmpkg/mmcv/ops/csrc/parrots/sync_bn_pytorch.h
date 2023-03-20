// Copyright (c) OpenMMLab. All rights reserved
#ifndef SYNC_BN_PYTORCH_H
#define SYNC_BN_PYTORCH_H
#include <torch/extension.h>
using namespace at;

void sync_bn_forward_mean_cuda(const Tensor input, Tensor mean);

void sync_bn_forward_var_cuda(const Tensor input, const Tensor mean,
                              Tensor var);

void sync_bn_forward_output_cuda(const Tensor input, const Tensor mean,
                                 const Tensor var, Tensor running_mean,
                                 Tensor running_var, const Tensor weight,
                                 const Tensor bias, Tensor norm, Tensor std,
                                 Tensor output, float eps, float momentum,
                                 int group_size);

void sync_bn_backward_param_cuda(const Tensor grad_output, const Tensor norm,
                                 Tensor grad_weight, Tensor grad_bias);

void sync_bn_backward_data_cuda(const Tensor grad_output, const Tensor weight,
                                const Tensor grad_weight,
                                const Tensor grad_bias, const Tensor norm,
                                const Tensor std, Tensor grad_input);
#endif  // SYNC_BN_PYTORCH_H
