// Copyright (c) OpenMMLab. All rights reserved.
#include <stdio.h>
#include <stdlib.h>
#include <torch/types.h>

#include "pytorch_cuda_helper.hpp"
#include "scatter_points_cuda_kernel.cuh"

std::vector<at::Tensor> DynamicPointToVoxelForwardCUDAKernelLauncher(
    const at::Tensor &feats, const at::Tensor &coors,
    const reduce_t reduce_type) {
  const int num_input = feats.size(0);
  const int num_feats = feats.size(1);

  if (num_input == 0)
    return {feats.clone().detach(), coors.clone().detach(),
            coors.new_empty({0}, torch::kInt32),
            coors.new_empty({0}, torch::kInt32)};

  at::Tensor out_coors;
  at::Tensor coors_map;
  at::Tensor reduce_count;

  auto coors_clean = coors.masked_fill(coors.lt(0).any(-1, true), -1);

  std::tie(out_coors, coors_map, reduce_count) =
      at::unique_dim(coors_clean, 0, true, true, true);

  if (out_coors[0][0].lt(0).item<bool>()) {
    // the first element of out_coors (-1,-1,-1) and should be removed
    out_coors = out_coors.slice(0, 1);
    reduce_count = reduce_count.slice(0, 1);
    coors_map = coors_map - 1;
  }

  coors_map = coors_map.to(torch::kInt32);
  reduce_count = reduce_count.to(torch::kInt32);

  auto reduced_feats =
      at::empty({out_coors.size(0), num_feats}, feats.options());

  at::cuda::CUDAGuard device_guard(feats.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(
      feats.scalar_type(), "feats_reduce_kernel", ([&] {
        if (reduce_type == reduce_t::MAX)
          reduced_feats.fill_(-std::numeric_limits<scalar_t>::infinity());
        else
          reduced_feats.fill_(static_cast<scalar_t>(0));

        dim3 blocks(std::min(
            at::cuda::ATenCeilDiv(num_input, THREADS_PER_BLOCK), maxGridDim));
        dim3 threads(THREADS_PER_BLOCK);
        feats_reduce_kernel<<<blocks, threads, 0, stream>>>(
            feats.data_ptr<scalar_t>(), coors_map.data_ptr<int32_t>(),
            reduced_feats.data_ptr<scalar_t>(), num_input, num_feats,
            reduce_type);
        if (reduce_type == reduce_t::MEAN)
          reduced_feats /= reduce_count.unsqueeze(-1).to(reduced_feats.dtype());
      }));

  AT_CUDA_CHECK(cudaGetLastError());

  return {reduced_feats, out_coors, coors_map, reduce_count};
}

void DynamicPointToVoxelBackwardCUDAKernelLauncher(
    at::Tensor &grad_feats, const at::Tensor &grad_reduced_feats,
    const at::Tensor &feats, const at::Tensor &reduced_feats,
    const at::Tensor &coors_map, const at::Tensor &reduce_count,
    const reduce_t reduce_type) {
  const int num_input = feats.size(0);
  const int num_reduced = reduced_feats.size(0);
  const int num_feats = feats.size(1);

  grad_feats.fill_(0);
  // copy voxel grad to points

  if (num_input == 0 || num_reduced == 0) return;
  at::cuda::CUDAGuard device_guard(feats.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (reduce_type == reduce_t::MEAN || reduce_type == reduce_t::SUM) {
    AT_DISPATCH_FLOATING_TYPES(
        grad_reduced_feats.scalar_type(), "add_reduce_traceback_grad_kernel",
        ([&] {
          dim3 blocks(std::min(
              at::cuda::ATenCeilDiv(num_input, THREADS_PER_BLOCK), maxGridDim));
          dim3 threads(THREADS_PER_BLOCK);
          add_reduce_traceback_grad_kernel<<<blocks, threads, 0, stream>>>(
              grad_feats.data_ptr<scalar_t>(),
              grad_reduced_feats.data_ptr<scalar_t>(),
              coors_map.data_ptr<int32_t>(), reduce_count.data_ptr<int32_t>(),
              num_input, num_feats, reduce_type);
        }));

    AT_CUDA_CHECK(cudaGetLastError());
  } else {
    auto reduce_from = at::full({num_reduced, num_feats}, num_input,
                                coors_map.options().dtype(torch::kInt32));
    AT_DISPATCH_FLOATING_TYPES(
        grad_reduced_feats.scalar_type(),
        "max_reduce_traceback_scatter_idx_kernel", ([&] {
          dim3 blocks(std::min(
              at::cuda::ATenCeilDiv(num_input, THREADS_PER_BLOCK), maxGridDim));
          dim3 threads(THREADS_PER_BLOCK);
          max_reduce_traceback_scatter_idx_kernel<<<blocks, threads, 0,
                                                    stream>>>(
              feats.data_ptr<scalar_t>(), reduced_feats.data_ptr<scalar_t>(),
              reduce_from.data_ptr<int32_t>(), coors_map.data_ptr<int32_t>(),
              num_input, num_feats);
        }));

    AT_CUDA_CHECK(cudaGetLastError());

    AT_DISPATCH_FLOATING_TYPES(
        grad_reduced_feats.scalar_type(),
        "max_reduce_traceback_scatter_idx_kernel", ([&] {
          dim3 blocks(
              std::min(at::cuda::ATenCeilDiv(num_reduced, THREADS_PER_BLOCK),
                       maxGridDim));
          dim3 threads(THREADS_PER_BLOCK);
          max_reduce_scatter_grad_kernel<<<blocks, threads, 0, stream>>>(
              grad_feats.data_ptr<scalar_t>(),
              grad_reduced_feats.data_ptr<scalar_t>(),
              reduce_from.data_ptr<int32_t>(), num_reduced, num_feats);
        }));

    AT_CUDA_CHECK(cudaGetLastError());
  }
}
