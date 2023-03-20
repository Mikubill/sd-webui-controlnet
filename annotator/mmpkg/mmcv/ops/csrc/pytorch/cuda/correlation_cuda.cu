// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/ClementPinard/Pytorch-Correlation-extension/blob/master/Correlation_Module/correlation_cuda_kernel.cu
// Original licence: Under MIT License

#include "correlation_cuda.cuh"
#include "pytorch_cuda_helper.hpp"

void CorrelationForwardCUDAKernelLauncher(Tensor input1, Tensor input2,
                                          Tensor output, int kH, int kW,
                                          int patchH, int patchW, int padH,
                                          int padW, int dilationH,
                                          int dilationW, int dilation_patchH,
                                          int dilation_patchW, int dH, int dW) {
  const int batch_size = input1.size(0);
  const int iH = input1.size(2);
  const int iW = input1.size(3);
  const int dilatedKH = (kH - 1) * dilationH + 1;
  const int dilatedKW = (kW - 1) * dilationW + 1;

  const auto oH = (iH + 2 * padH - dilatedKH) / dH + 1;
  const auto oW = (iW + 2 * padW - dilatedKW) / dW + 1;

  auto trInput1 = input1.permute({0, 2, 3, 1}).contiguous();
  auto trInput2 = input2.permute({0, 2, 3, 1}).contiguous();

  const dim3 threads(WARP_SIZE, 4, 4);
  const dim3 blocks(batch_size, (oH + 3) >> 2, (oW + 3) >> 2);

  at::cuda::CUDAGuard device_guard(input1.device());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input1.scalar_type(), "correlation_forward_cuda", ([&] {
        TensorAcc4R trInput1_acc =
            trInput1.packed_accessor32<scalar_t, 4, RestrictPtrTraits>();
        TensorAcc4R trInput2_acc =
            trInput2.packed_accessor32<scalar_t, 4, RestrictPtrTraits>();
        TensorAcc5R output_acc =
            output.packed_accessor32<scalar_t, 5, RestrictPtrTraits>();

        correlation_forward_cuda_kernel<scalar_t>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                trInput1_acc, trInput2_acc, output_acc, kH, kW, patchH, patchW,
                padH, padW, dilationH, dilationW, dilation_patchH,
                dilation_patchW, dH, dW, oH, oW);
      }));
}

void CorrelationBackwardCUDAKernelLauncher(
    Tensor grad_output, Tensor input1, Tensor input2, Tensor grad_input1,
    Tensor grad_input2, int kH, int kW, int patchH, int patchW, int padH,
    int padW, int dilationH, int dilationW, int dilation_patchH,
    int dilation_patchW, int dH, int dW) {
  const int batch_size = input1.size(0);
  const int iH = input1.size(2);
  const int iW = input1.size(3);
  const int C = input1.size(1);

  auto trInput1 = input1.permute({0, 2, 3, 1}).contiguous();
  auto trInput2 = input2.permute({0, 2, 3, 1}).contiguous();
  const dim3 blocks(batch_size, iH, iW);
  const dim3 threads(THREADS_PER_BLOCK);

  at::cuda::CUDAGuard device_guard(input1.device());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input1.scalar_type(), "correlation_backward_cuda", ([&] {
        const int grad_cache_size = patchH * patchW * sizeof(scalar_t);
        TensorAcc4R input1_acc =
            trInput1.packed_accessor32<scalar_t, 4, RestrictPtrTraits>();
        TensorAcc4R input2_acc =
            trInput2.packed_accessor32<scalar_t, 4, RestrictPtrTraits>();
        TensorAcc4R grad_input1_acc =
            grad_input1.packed_accessor32<scalar_t, 4, RestrictPtrTraits>();
        TensorAcc4R grad_input2_acc =
            grad_input2.packed_accessor32<scalar_t, 4, RestrictPtrTraits>();
        TensorAcc5R grad_output_acc =
            grad_output.packed_accessor32<scalar_t, 5, RestrictPtrTraits>();

        correlation_backward_cuda_kernel_input1<scalar_t>
            <<<blocks, threads, grad_cache_size,
               at::cuda::getCurrentCUDAStream()>>>(
                grad_output_acc, input2_acc, grad_input1_acc, kH, kW, patchH,
                patchW, padH, padW, dilationH, dilationW, dilation_patchH,
                dilation_patchW, dH, dW);

        correlation_backward_cuda_kernel_input2<scalar_t>
            <<<blocks, threads, grad_cache_size,
               at::cuda::getCurrentCUDAStream()>>>(
                grad_output_acc, input1_acc, grad_input2_acc, kH, kW, patchH,
                patchW, padH, padW, dilationH, dilationW, dilation_patchH,
                dilation_patchW, dH, dW);
      }));
}
