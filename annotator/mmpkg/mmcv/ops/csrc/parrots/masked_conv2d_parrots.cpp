// Copyright (c) OpenMMLab. All rights reserved
#include <parrots/compute/aten.hpp>
#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>

#include "masked_conv2d_pytorch.h"

using namespace parrots;

#ifdef MMCV_WITH_CUDA
void masked_im2col_forward_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                        const OperatorBase::in_list_t& ins,
                                        OperatorBase::out_list_t& outs) {
  // im: (n, ic, h, w), kernel size (kh, kw)
  // kernel: (oc, ic * kh * kw), col: (kh * kw * ic, ow * oh)
  int kernel_h, kernel_w, pad_h, pad_w;
  SSAttrs(attr)
      .get<int>("kernel_h", kernel_h)
      .get<int>("kernel_w", kernel_w)
      .get<int>("pad_h", pad_h)
      .get<int>("pad_w", pad_w)
      .done();

  const auto& im = buildATensor(ctx, ins[0]);
  const auto& mask_h_idx = buildATensor(ctx, ins[1]);
  const auto& mask_w_idx = buildATensor(ctx, ins[2]);

  auto col = buildATensor(ctx, outs[0]);
  masked_im2col_forward_cuda(im, mask_h_idx, mask_w_idx, col, kernel_h,
                             kernel_w, pad_h, pad_w);
}

void masked_col2im_forward_cuda_parrots(CudaContext& ctx, const SSElement& attr,
                                        const OperatorBase::in_list_t& ins,
                                        OperatorBase::out_list_t& outs) {
  // im: (n, ic, h, w), kernel size (kh, kw)
  // kernel: (oc, ic * kh * kh), col: (kh * kw * ic, ow * oh)
  int height, width, channels;
  SSAttrs(attr)
      .get<int>("height", height)
      .get<int>("width", width)
      .get<int>("channels", channels)
      .done();

  const auto& col = buildATensor(ctx, ins[0]);
  const auto& mask_h_idx = buildATensor(ctx, ins[1]);
  const auto& mask_w_idx = buildATensor(ctx, ins[2]);

  auto im = buildATensor(ctx, outs[0]);
  masked_col2im_forward_cuda(col, mask_h_idx, mask_w_idx, im, height, width,
                             channels);
}

PARROTS_EXTENSION_REGISTER(masked_im2col_forward)
    .attr("kernel_h")
    .attr("kernel_w")
    .attr("pad_h")
    .attr("pad_w")
    .input(3)
    .output(1)
    .apply(masked_im2col_forward_cuda_parrots)
    .done();

PARROTS_EXTENSION_REGISTER(masked_col2im_forward)
    .attr("height")
    .attr("width")
    .attr("channels")
    .input(3)
    .output(1)
    .apply(masked_col2im_forward_cuda_parrots)
    .done();
#endif
