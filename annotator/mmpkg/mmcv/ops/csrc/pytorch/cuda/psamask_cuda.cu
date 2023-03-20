// Copyright (c) OpenMMLab. All rights reserved
// Modified from
// https://github.com/hszhao/semseg/blob/master/lib/psa/src

#include <torch/serialize/tensor.h>

#include "psamask_cuda_kernel.cuh"
#include "pytorch_cuda_helper.hpp"

void PSAMaskForwardCUDAKernelLauncher(const int psa_type, const Tensor input,
                                      Tensor output, const int num_,
                                      const int h_feature, const int w_feature,
                                      const int h_mask, const int w_mask,
                                      const int half_h_mask,
                                      const int half_w_mask) {
  int nthreads = num_ * h_feature * w_feature;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (psa_type == 0)
    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "psamask_collect_forward_cuda", [&] {
          psamask_collect_forward_cuda<scalar_t><<<nthreads, 512, 0, stream>>>(
              nthreads, h_feature, w_feature, h_mask, w_mask, half_h_mask,
              half_w_mask, input.data_ptr<scalar_t>(),
              output.data_ptr<scalar_t>());
        });
  else
    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "psamask_distribute_forward_cuda", [&] {
          psamask_distribute_forward_cuda<scalar_t>
              <<<nthreads, 512, 0, stream>>>(
                  nthreads, h_feature, w_feature, h_mask, w_mask, half_h_mask,
                  half_w_mask, input.data_ptr<scalar_t>(),
                  output.data_ptr<scalar_t>());
        });
}

void PSAMaskBackwardCUDAKernelLauncher(
    const int psa_type, const Tensor grad_output, Tensor grad_input,
    const int num_, const int h_feature, const int w_feature, const int h_mask,
    const int w_mask, const int half_h_mask, const int half_w_mask) {
  int nthreads = num_ * h_feature * w_feature;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (psa_type == 0)
    AT_DISPATCH_FLOATING_TYPES(
        grad_input.scalar_type(), "psamask_collect_backward_cuda", [&] {
          psamask_collect_backward_cuda<scalar_t><<<nthreads, 512, 0, stream>>>(
              nthreads, h_feature, w_feature, h_mask, w_mask, half_h_mask,
              half_w_mask, grad_output.data_ptr<scalar_t>(),
              grad_input.data_ptr<scalar_t>());
        });
  else
    AT_DISPATCH_FLOATING_TYPES(
        grad_input.scalar_type(), "psamask_distribute_backward_cuda", [&] {
          psamask_distribute_backward_cuda<scalar_t>
              <<<nthreads, 512, 0, stream>>>(
                  nthreads, h_feature, w_feature, h_mask, w_mask, half_h_mask,
                  half_w_mask, grad_output.data_ptr<scalar_t>(),
                  grad_input.data_ptr<scalar_t>());
        });
}
