// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "carafe_naive_pytorch.h"

using namespace parrots;

#ifdef MMCV_WITH_CUDA
/*void carafe_naive_forward_cuda(Tensor features, Tensor masks, Tensor output,
 *                                int kernel_size, int group_size,
 *                                int scale_factor)
 */
void carafe_naive_forward_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                       const OperatorBase::in_list_t& ins,
                                       OperatorBase::out_list_t& outs) {
  int kernel_size, group_size, scale_factor;
  SSAttrs(attr)
      .get<int>("kernel_size", kernel_size)
      .get<int>("group_size", group_size)
      .get<int>("scale_factor", scale_factor)
      .done();

  const auto& features = buildATensor(ctx, ins[0]);
  const auto& masks = buildATensor(ctx, ins[1]);

  auto output = buildATensor(ctx, outs[0]);
  carafe_naive_forward_cuda(features, masks, output, kernel_size, group_size,
                            scale_factor);
}

/*void carafe_naive_backward_cuda(Tensor top_grad, Tensor features, Tensor
 * masks, Tensor bottom_grad, Tensor mask_grad, int kernel_size, int group_size,
 *                                int scale_factor);
 */
void carafe_naive_backward_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                        const OperatorBase::in_list_t& ins,
                                        OperatorBase::out_list_t& outs) {
  int kernel_size, group_size, scale_factor;
  SSAttrs(attr)
      .get<int>("kernel_size", kernel_size)
      .get<int>("group_size", group_size)
      .get<int>("scale_factor", scale_factor)
      .done();

  const auto& top_grad = buildATensor(ctx, ins[0]);
  const auto& features = buildATensor(ctx, ins[1]);
  const auto& masks = buildATensor(ctx, ins[2]);

  auto bottom_grad = buildATensor(ctx, outs[0]);
  auto mask_grad = buildATensor(ctx, outs[1]);
  carafe_naive_backward_cuda(top_grad, features, masks, bottom_grad, mask_grad,
                             kernel_size, group_size, scale_factor);
}

PARROTS_EXTENSION_REGISTER(carafe_naive_forward)
    .attr("kernel_size")
    .attr("group_size")
    .attr("scale_factor")
    .input(2)
    .output(1)
    .apply(carafe_naive_forward_cuda_parrots)
    .done();

PARROTS_EXTENSION_REGISTER(carafe_naive_backward)
    .attr("kernel_size")
    .attr("group_size")
    .attr("scale_factor")
    .input(3)
    .output(2)
    .apply(carafe_naive_backward_cuda_parrots)
    .done();
#endif
