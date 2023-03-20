// Copyright (c) OpenMMLab. All rights reserved
#include "reduce_ops.h"

#include <assert.h>

#include <vector>

#include "../ort_mmcv_utils.h"

// modified from
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/ReduceOps.cpp

static inline int64_t maybe_wrap_dim(int64_t dim, int64_t ndims) {
  int64_t min = -ndims;
  int64_t max = ndims - 1;
  assert(dim >= min && dim <= max);
  if (dim < 0) dim += ndims;
  return dim;
}

static inline int64_t get_dim_stride(const int64_t dim, const int64_t ndims,
                                     const int64_t *reversed_dim_cumprod) {
  return dim == ndims - 1 ? 1 : reversed_dim_cumprod[dim + 1];
}

static inline int64_t get_dim_size(const int64_t dim, const int64_t ndims,
                                   const int64_t *reversed_dim_cumprod) {
  return dim == ndims - 1
             ? reversed_dim_cumprod[dim]
             : reversed_dim_cumprod[dim] / reversed_dim_cumprod[dim + 1];
}

template <typename T1, typename T2, typename Operation>
void cummax_cummin_helper(const T1 *input, T1 *output, T2 *indices,
                          const int64_t input_dim_size, const int64_t stride) {
  Operation op;
  T1 out = input[0];
  int64_t idx = 0;
  for (int64_t i = 0; i < input_dim_size; i++) {
    T1 curr_elem = input[i * stride];
    if (op(curr_elem, out)) {
      out = curr_elem;
      idx = i;
    }
    output[i * stride] = out;
    indices[i * stride] = idx;
  }
}

// modified `tensor_dim_apply3` from
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/TensorDimApply.h.
// the difference is that: (1) use `reversed_dim_cumprod` for fast computing of
// tensor `size` and `stride`. (2) the same `stride` is used for input, output,
// and indices, since it's unnecessary to use separate values. currently
// `tensor_dim_apply3` is only used for `cummax` and `cummin`, according to the
// official pytorch projects: https://github.com/pytorch/pytorch.
template <typename T1, typename T2, typename Function>
void tensor_dim_apply3(const T1 *input, T1 *output, T2 *indices,
                       const int64_t dim, const int64_t ndims,
                       const int64_t *reversed_dim_cumprod, Function func) {
  int dim_apply_finished = 0;
  int64_t input_dim_size = get_dim_size(dim, ndims, reversed_dim_cumprod);
  // the same stride is used for input, output and indices
  int64_t stride = get_dim_stride(dim, ndims, reversed_dim_cumprod);
  std::vector<int64_t> counter(ndims, 0);

  while (!dim_apply_finished) {
    // call `func` once to update output and indices
    func(input, output, indices, input_dim_size, stride);
    if (ndims == 1) break;
    for (int64_t dim_i = 0; dim_i < ndims; dim_i++) {
      if (dim_i == dim) {
        if (dim_i == (ndims - 1)) {
          dim_apply_finished = 1;
          break;
        }
        continue;
      }
      counter[dim_i]++;

      // the same stride is used for input, output, and indices
      int64_t stride_dim_i = get_dim_stride(dim_i, ndims, reversed_dim_cumprod);
      input += stride_dim_i;
      output += stride_dim_i;
      indices += stride_dim_i;

      if (counter[dim_i] == get_dim_size(dim_i, ndims, reversed_dim_cumprod)) {
        if (dim_i == ndims - 1) {
          dim_apply_finished = 1;
          break;
        } else {
          input -= counter[dim_i] * stride_dim_i;
          output -= counter[dim_i] * stride_dim_i;
          indices -= counter[dim_i] * stride_dim_i;
          counter[dim_i] = 0;
        }
      } else {
        break;
      }  // if
    }    // for
  }      // while
}

template <typename T1, typename T2, typename Operation>
void CumMax_CumMin_CPU(const T1 *input, T1 *output, T2 *indices,
                       int64_t *reversed_dim_cumprod, const int64_t dim,
                       const OrtTensorDimensions &out_dimensions) {
  // calculate numel
  const int64_t ndims = out_dimensions.size();
  int64_t numel = 1;
  for (int64_t dim_i = 0; dim_i < ndims; dim_i++) {
    numel *= out_dimensions.data()[dim_i];
  }

  // cummax is only applied to input which is non-zero dim and non-empty
  if (numel) {
    // compute the cumulative production on dimension size,
    // which is then used for computing the stride or size of a specific `dim`.
    reversed_dim_cumprod[ndims - 1] = out_dimensions.data()[ndims - 1];
    for (int64_t dim_i = ndims - 2; dim_i >= 0; dim_i--) {
      reversed_dim_cumprod[dim_i] =
          reversed_dim_cumprod[dim_i + 1] * out_dimensions.data()[dim_i];
    }

    // do cummax or cummin based on `Operation` type
    tensor_dim_apply3<float, int64_t>(
        input, output, indices, dim, ndims, reversed_dim_cumprod,
        cummax_cummin_helper<float, int64_t, Operation>);
  }
}

void MMCVCumMaxKernel::Compute(OrtKernelContext *context) {
  // get input
  const OrtValue *input = ort_.KernelContext_GetInput(context, 0);
  const float *input_data =
      reinterpret_cast<const float *>(ort_.GetTensorData<float>(input));

  // get output
  OrtTensorDimensions out_dimensions(ort_, input);
  OrtValue *output = ort_.KernelContext_GetOutput(
      context, 0, out_dimensions.data(), out_dimensions.size());
  float *output_data = ort_.GetTensorMutableData<float>(output);
  OrtValue *indices = ort_.KernelContext_GetOutput(
      context, 1, out_dimensions.data(), out_dimensions.size());
  int64_t *indices_data = ort_.GetTensorMutableData<int64_t>(indices);

  // allocate tmp memory for computing the cumulative production on dimension
  // size
  const int64_t ndims = out_dimensions.size();
  assert(ndims > 0);
  int64_t *reversed_dim_cumprod =
      (int64_t *)allocator_.Alloc(sizeof(int64_t) * ndims);

  // dim should be wrapped if it's negative (e.g. -1)
  const int64_t dim = maybe_wrap_dim(dim_, ndims);
  CumMax_CumMin_CPU<float, int64_t, std::greater_equal<float>>(
      input_data, output_data, indices_data, reversed_dim_cumprod, dim,
      out_dimensions);
}

void MMCVCumMinKernel::Compute(OrtKernelContext *context) {
  // get input
  const OrtValue *input = ort_.KernelContext_GetInput(context, 0);
  const float *input_data =
      reinterpret_cast<const float *>(ort_.GetTensorData<float>(input));

  // get output
  OrtTensorDimensions out_dimensions(ort_, input);
  OrtValue *output = ort_.KernelContext_GetOutput(
      context, 0, out_dimensions.data(), out_dimensions.size());
  float *output_data = ort_.GetTensorMutableData<float>(output);
  OrtValue *indices = ort_.KernelContext_GetOutput(
      context, 1, out_dimensions.data(), out_dimensions.size());
  int64_t *indices_data = ort_.GetTensorMutableData<int64_t>(indices);

  // allocate tmp memory for computing the cumulative production on dimension
  // size
  const int64_t ndims = out_dimensions.size();
  assert(ndims > 0);
  int64_t *reversed_dim_cumprod =
      (int64_t *)allocator_.Alloc(sizeof(int64_t) * ndims);

  // dim should be wrapped if it's negative (e.g. -1)
  const int64_t dim = maybe_wrap_dim(dim_, ndims);
  CumMax_CumMin_CPU<float, int64_t, std::less_equal<float>>(
      input_data, output_data, indices_data, reversed_dim_cumprod, dim,
      out_dimensions);
}
