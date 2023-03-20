// Copyright (c) OpenMMLab. All rights reserved
#include "carafe_cuda_kernel.cuh"
#include "pytorch_cuda_helper.hpp"

void CARAFEForwardCUDAKernelLauncher(const Tensor features, const Tensor masks,
                                     Tensor rfeatures, Tensor routput,
                                     Tensor rmasks, Tensor output,
                                     const int kernel_size,
                                     const int group_size,
                                     const int scale_factor) {
  const int batch_size = output.size(0);
  const int channels = output.size(1);
  const int output_height = output.size(2);
  const int output_width = output.size(3);

  const int input_height = features.size(2);
  const int input_width = features.size(3);

  const int mask_channels = masks.size(1);

  rfeatures.resize_({batch_size, input_height, input_width, channels});
  routput.resize_({batch_size, output_height, output_width, channels});
  rmasks.resize_({batch_size, output_height, output_width, mask_channels});

  // one warp per pixel
  at::cuda::CUDAGuard device_guard(features.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      features.scalar_type(), "NCHW2NHWC_Feature", ([&] {
        const scalar_t *bottom_data = features.data_ptr<scalar_t>();
        scalar_t *top_data = rfeatures.data_ptr<scalar_t>();
        const int dh = divideUP(channels, kTileDim);
        const int dw = divideUP(input_height * input_width, kTileDim);
        BatchTranspose2DCUDAKernel<scalar_t>
            <<<batch_size * dh * dw, dim3(kTileDim, kBlockRows), 0, stream>>>(
                batch_size, channels, input_height * input_width, dh, dw,
                bottom_data, top_data);
      }));
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      features.scalar_type(), "NCHW2NHWC_Masks", ([&] {
        const scalar_t *bottom_data = masks.data_ptr<scalar_t>();
        scalar_t *top_data = rmasks.data_ptr<scalar_t>();
        const int dh = divideUP(mask_channels, kTileDim);
        const int dw = divideUP(output_height * output_width, kTileDim);
        BatchTranspose2DCUDAKernel<scalar_t>
            <<<batch_size * dh * dw, dim3(kTileDim, kBlockRows), 0, stream>>>(
                batch_size, mask_channels, output_height * output_width, dh, dw,
                bottom_data, top_data);
      }));
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      features.scalar_type(), "CARAFELaucherForward", ([&] {
        const int num_kernels =
            batch_size * output_height * output_width * THREADS_PER_PIXEL;
        const scalar_t *bottom_data = rfeatures.data_ptr<scalar_t>();
        const scalar_t *bottom_masks = rmasks.data_ptr<scalar_t>();
        scalar_t *top_data = routput.data_ptr<scalar_t>();

        CARAFEForward<scalar_t><<<divideUP(num_kernels, THREADS_PER_BLOCK),
                                  THREADS_PER_BLOCK, 0, stream>>>(
            num_kernels, bottom_data, bottom_masks, kernel_size, group_size,
            scale_factor, channels, input_height, input_width, output_height,
            output_width, mask_channels, top_data);
      }));
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      features.scalar_type(), "NHWC2NCHW", ([&] {
        const scalar_t *bottom_data = routput.data_ptr<scalar_t>();
        scalar_t *top_data = output.data_ptr<scalar_t>();
        const int dh = divideUP(output_height * output_width, kTileDim);
        const int dw = divideUP(channels, kTileDim);
        BatchTranspose2DCUDAKernel<scalar_t>
            <<<batch_size * dh * dw, dim3(kTileDim, kBlockRows), 0, stream>>>(
                batch_size, output_height * output_width, channels, dh, dw,
                bottom_data, top_data);
      }));

  AT_CUDA_CHECK(cudaGetLastError());
}

void CARAFEBackwardCUDAKernelLauncher(
    const Tensor top_grad, const Tensor rfeatures, const Tensor masks,
    Tensor rtop_grad, Tensor rbottom_grad_hs, Tensor rbottom_grad,
    Tensor rmask_grad, Tensor bottom_grad, Tensor mask_grad,
    const int kernel_size, const int group_size, const int scale_factor) {
  const int batch_size = top_grad.size(0);
  const int channels = top_grad.size(1);
  const int output_height = top_grad.size(2);
  const int output_width = top_grad.size(3);

  const int input_height = bottom_grad.size(2);
  const int input_width = bottom_grad.size(3);

  const int mask_channels = masks.size(1);

  rtop_grad.resize_({batch_size, output_height, output_width, channels});
  rbottom_grad.resize_({batch_size, input_height, input_width, channels});
  rbottom_grad_hs.resize_({batch_size, output_height, output_width, channels});
  rmask_grad.resize_({batch_size, output_height, output_width, mask_channels});

  at::cuda::CUDAGuard device_guard(top_grad.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.scalar_type(), "NCHW2NHWC_Top_Grad", ([&] {
        const scalar_t *bottom_data = top_grad.data_ptr<scalar_t>();
        scalar_t *top_data = rtop_grad.data_ptr<scalar_t>();
        const int dh = divideUP(channels, kTileDim);
        const int dw = divideUP(output_height * output_width, kTileDim);
        BatchTranspose2DCUDAKernel<scalar_t>
            <<<batch_size * dh * dw, dim3(kTileDim, kBlockRows), 0, stream>>>(
                batch_size, channels, output_height * output_width, dh, dw,
                bottom_data, top_data);
      }));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.scalar_type(), "CARAFELaucherBackward_Feature", ([&] {
        const int num_kernels =
            batch_size * output_height * output_width * THREADS_PER_PIXEL;
        const scalar_t *top_diff = rtop_grad.data_ptr<scalar_t>();
        const scalar_t *bottom_masks = masks.data_ptr<scalar_t>();
        scalar_t *bottom_diff = rbottom_grad_hs.data_ptr<scalar_t>();

        CARAFEBackward_Feature<scalar_t>
            <<<divideUP(num_kernels, THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0,
               stream>>>(num_kernels, top_diff, bottom_masks, kernel_size,
                         group_size, scale_factor, channels, input_height,
                         input_width, output_height, output_width,
                         mask_channels, bottom_diff);
      }));
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.scalar_type(), "FeatureSum", ([&] {
        const int num_kernels =
            batch_size * input_height * input_width * THREADS_PER_PIXEL;
        const scalar_t *bottom_diff_hs = rbottom_grad_hs.data_ptr<scalar_t>();
        scalar_t *bottom_diff = rbottom_grad.data_ptr<scalar_t>();

        FeatureSum<scalar_t>
            <<<divideUP(num_kernels, THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0,
               stream>>>(num_kernels, bottom_diff_hs, scale_factor, channels,
                         input_height, input_width, bottom_diff);
      }));
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.scalar_type(), "NHWC2NCHW_Bottom_Grad", ([&] {
        const scalar_t *bottom_data = rbottom_grad.data_ptr<scalar_t>();
        scalar_t *top_data = bottom_grad.data_ptr<scalar_t>();
        const int dh = divideUP(input_height * input_width, kTileDim);
        const int dw = divideUP(channels, kTileDim);
        BatchTranspose2DCUDAKernel<scalar_t>
            <<<batch_size * dh * dw, dim3(kTileDim, kBlockRows), 0, stream>>>(
                batch_size, input_height * input_width, channels, dh, dw,
                bottom_data, top_data);
      }));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.scalar_type(), "CARAFELaucherBackward_Mask", ([&] {
        const int num_kernels = batch_size * output_height * output_width *
                                mask_channels * WARP_SIZE;
        const scalar_t *top_diff = rtop_grad.data_ptr<scalar_t>();
        const scalar_t *bottom_data = rfeatures.data_ptr<scalar_t>();
        scalar_t *mask_diff = rmask_grad.data_ptr<scalar_t>();

        CARAFEBackward_Mask<scalar_t>
            <<<divideUP(num_kernels, THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0,
               stream>>>(num_kernels, top_diff, bottom_data, kernel_size,
                         group_size, scale_factor, channels, input_height,
                         input_width, output_height, output_width,
                         mask_channels, mask_diff);
      }));
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.scalar_type(), "NHWC2NCHW_Mask_Grad", ([&] {
        const scalar_t *bottom_data = rmask_grad.data_ptr<scalar_t>();
        scalar_t *top_data = mask_grad.data_ptr<scalar_t>();
        const int dh = divideUP(output_height * output_width, kTileDim);
        const int dw = divideUP(mask_channels, kTileDim);
        BatchTranspose2DCUDAKernel<scalar_t>
            <<<batch_size * dh * dw, dim3(kTileDim, kBlockRows), 0, stream>>>(
                batch_size, output_height * output_width, mask_channels, dh, dw,
                bottom_data, top_data);
      }));

  AT_CUDA_CHECK(cudaGetLastError());
}
