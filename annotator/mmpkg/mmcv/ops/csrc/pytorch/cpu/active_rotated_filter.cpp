// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
// modified from
// https://github.com/csuhan/s2anet/blob/master/mmdet/ops/orn/src/cpu/ActiveRotatingFilter_cpu.cpp
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

template <typename T>
void active_rotated_filter_forward_cpu_kernel(
    const T* weightData, const int* indicesData, const int num_output_planes,
    const int num_input_planes, const int num_orientations, const int kH,
    const int kW, const int num_rotations, T* outputData) {
  const int nEntry = num_orientations * kH * kW;
  int i, j, l;
  int k;

#pragma omp parallel for private(i, j, l, k)
  for (i = 0; i < num_output_planes; i++) {
    for (j = 0; j < num_input_planes; j++) {
      for (l = 0; l < nEntry; l++) {
        int weightIndex = i * num_input_planes * nEntry + j * nEntry + l;
        T val = *(weightData + weightIndex);
        for (k = 0; k < num_rotations; k++) {
          int index = (int)(*(indicesData + l * num_rotations + k)) - 1;
          T* target = outputData +
                      i * (num_rotations * num_input_planes * nEntry) +
                      k * (num_input_planes * nEntry) + j * (nEntry) + index;
          *target = val;
        }
      }
    }
  }
}

template <typename T>
void active_rotated_filter_backward_cpu_kernel(
    const T* gradOutputData, const int* indicesData,
    const int num_output_planes, const int num_input_planes,
    const int num_orientations, const int kH, const int kW,
    const int num_rotations, T* gradInputData) {
  const int nEntry = num_orientations * kH * kW;
  int i, j, l;
  int k;

#pragma omp parallel for private(i, j, l, k)
  for (i = 0; i < num_output_planes; i++) {
    for (j = 0; j < num_input_planes; j++) {
      for (l = 0; l < nEntry; l++) {
        int gradInputIndex = i * num_input_planes * nEntry + j * nEntry + l;
        T* val = gradInputData + gradInputIndex;
        *val = 0;
        for (k = 0; k < num_rotations; k++) {
          int index = (int)(*(indicesData + l * num_rotations + k)) - 1;
          const T* target =
              gradOutputData + i * (num_rotations * num_input_planes * nEntry) +
              k * (num_input_planes * nEntry) + j * (nEntry) + index;
          *val = *val + *target;
        }
      }
    }
  }
}

void ActiveRotatedFilterForwardCPULauncher(const Tensor input,
                                           const Tensor indices,
                                           Tensor output) {
  const int num_output_planes = input.size(0);
  const int num_input_planes = input.size(1);
  const int num_orientations = input.size(2);
  const int kH = input.size(3);
  const int kW = input.size(4);
  const int num_rotations = indices.size(3);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "active_rotated_filter_forward_cpu_kernel", [&] {
        active_rotated_filter_forward_cpu_kernel<scalar_t>(
            input.data_ptr<scalar_t>(), indices.data_ptr<int>(),
            num_output_planes, num_input_planes, num_orientations, kH, kW,
            num_rotations, output.data_ptr<scalar_t>());
      });
}

void ActiveRotatedFilterBackwardCPULauncher(const Tensor grad_out,
                                            const Tensor indices,
                                            Tensor grad_in) {
  const int num_orientations = indices.size(0);
  const int kH = indices.size(1);
  const int kW = indices.size(2);
  const int num_rotations = indices.size(3);
  const int num_output_planes = grad_out.size(0) / num_rotations;
  const int num_input_planes = grad_out.size(1) / num_orientations;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_out.scalar_type(), "active_rotated_filter_backward_cpu_kernel", [&] {
        active_rotated_filter_backward_cpu_kernel<scalar_t>(
            grad_out.data_ptr<scalar_t>(), indices.data_ptr<int>(),
            num_output_planes, num_input_planes, num_orientations, kH, kW,
            num_rotations, grad_in.data_ptr<scalar_t>());
      });
}

void active_rotated_filter_forward_cpu(const Tensor input, const Tensor indices,
                                       Tensor output) {
  ActiveRotatedFilterForwardCPULauncher(input, indices, output);
}

void active_rotated_filter_backward_cpu(const Tensor grad_out,
                                        const Tensor indices, Tensor grad_in) {
  ActiveRotatedFilterBackwardCPULauncher(grad_out, indices, grad_in);
}

void active_rotated_filter_forward_impl(const Tensor input,
                                        const Tensor indices, Tensor output);

void active_rotated_filter_backward_impl(const Tensor grad_out,
                                         const Tensor indices, Tensor grad_in);

REGISTER_DEVICE_IMPL(active_rotated_filter_forward_impl, CPU,
                     active_rotated_filter_forward_cpu);
REGISTER_DEVICE_IMPL(active_rotated_filter_backward_impl, CPU,
                     active_rotated_filter_backward_cpu);
