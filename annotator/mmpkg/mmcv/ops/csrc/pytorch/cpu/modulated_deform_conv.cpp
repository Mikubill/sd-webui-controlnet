// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

template <typename T>
T dmcn_im2col_bilinear_cpu(const T *input, const int data_width,
                           const int height, const int width, T h, T w) {
  int h_low = floorf(h);
  int w_low = floorf(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  T lh = h - h_low;
  T lw = w - w_low;
  T hh = 1 - lh, hw = 1 - lw;

  T v1 = 0;
  if (h_low >= 0 && w_low >= 0) v1 = input[h_low * data_width + w_low];
  T v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = input[h_low * data_width + w_high];
  T v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = input[h_high * data_width + w_low];
  T v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = input[h_high * data_width + w_high];

  T w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename T>
T dmcn_get_gradient_weight_cpu(T argmax_h, T argmax_w, const int h, const int w,
                               const int height, const int width) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 ||
      argmax_w >= width) {
    // empty
    return 0;
  }

  int argmax_h_low = floorf(argmax_h);
  int argmax_w_low = floorf(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  T weight = 0;
  if (h == argmax_h_low && w == argmax_w_low)
    weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
  if (h == argmax_h_low && w == argmax_w_high)
    weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
  if (h == argmax_h_high && w == argmax_w_low)
    weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
  if (h == argmax_h_high && w == argmax_w_high)
    weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
  return weight;
}

template <typename T>
T dmcn_get_coordinate_weight_cpu(T argmax_h, T argmax_w, const int height,
                                 const int width, const T *im_data,
                                 const int data_width, const int bp_dir) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 ||
      argmax_w >= width) {
    // empty
    return 0;
  }

  int argmax_h_low = floorf(argmax_h);
  int argmax_w_low = floorf(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  T weight = 0;

  if (bp_dir == 0) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_w_low + 1 - argmax_w) *
                im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += -1 * (argmax_w - argmax_w_low) *
                im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += (argmax_w_low + 1 - argmax_w) *
                im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_w - argmax_w_low) *
                im_data[argmax_h_high * data_width + argmax_w_high];
  } else if (bp_dir == 1) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_h_low + 1 - argmax_h) *
                im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += (argmax_h_low + 1 - argmax_h) *
                im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += -1 * (argmax_h - argmax_h_low) *
                im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_h - argmax_h_low) *
                im_data[argmax_h_high * data_width + argmax_w_high];
  }

  return weight;
}

template <typename T>
void modulated_deformable_im2col_cpu_kernel(
    const int n, const T *data_im, const T *data_offset, const T *data_mask,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group, const int batch_size,
    const int num_channels, const int deformable_group, const int height_col,
    const int width_col, T *data_col) {
  for (int index = 0; index < n; index++) {
    // index index of output matrix
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int b_col = (index / width_col / height_col) % batch_size;
    const int c_im = (index / width_col / height_col) / batch_size;
    const int c_col = c_im * kernel_h * kernel_w;

    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;

    T *data_col_ptr =
        data_col +
        ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    const T *data_im_ptr =
        data_im + (b_col * num_channels + c_im) * height * width;
    const T *data_offset_ptr =
        data_offset + (b_col * deformable_group + deformable_group_index) * 2 *
                          kernel_h * kernel_w * height_col * width_col;

    const T *data_mask_ptr =
        data_mask + (b_col * deformable_group + deformable_group_index) *
                        kernel_h * kernel_w * height_col * width_col;

    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        const int data_offset_h_ptr =
            ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr =
            ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col +
            w_col;
        const int data_mask_hw_ptr =
            ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;
        const T offset_h = data_offset_ptr[data_offset_h_ptr];
        const T offset_w = data_offset_ptr[data_offset_w_ptr];
        const T mask = data_mask_ptr[data_mask_hw_ptr];
        T val = static_cast<T>(0);
        const T h_im = h_in + i * dilation_h + offset_h;
        const T w_im = w_in + j * dilation_w + offset_w;
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width)
          val = dmcn_im2col_bilinear_cpu(data_im_ptr, width, height, width,
                                         h_im, w_im);
        *data_col_ptr = val * mask;
        data_col_ptr += batch_size * height_col * width_col;
      }
    }
  }
}

template <typename T>
void modulated_deformable_col2im_cpu_kernel(
    const int n, const T *data_col, const T *data_offset, const T *data_mask,
    const int channels, const int height, const int width, const int kernel_h,
    const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group, const int batch_size,
    const int deformable_group, const int height_col, const int width_col,
    T *grad_im) {
  for (int index = 0; index < n; index++) {
    const int j = (index / width_col / height_col / batch_size) % kernel_w;
    const int i =
        (index / width_col / height_col / batch_size / kernel_w) % kernel_h;
    const int c =
        index / width_col / height_col / batch_size / kernel_w / kernel_h;
    // compute the start and end of the output

    const int deformable_group_index = c / channel_per_deformable_group;

    int w_out = index % width_col;
    int h_out = (index / width_col) % height_col;
    int b = (index / width_col / height_col) % batch_size;
    int w_in = w_out * stride_w - pad_w;
    int h_in = h_out * stride_h - pad_h;

    const T *data_offset_ptr =
        data_offset + (b * deformable_group + deformable_group_index) * 2 *
                          kernel_h * kernel_w * height_col * width_col;
    const T *data_mask_ptr =
        data_mask + (b * deformable_group + deformable_group_index) * kernel_h *
                        kernel_w * height_col * width_col;
    const int data_offset_h_ptr =
        ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr =
        ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
    const int data_mask_hw_ptr =
        ((i * kernel_w + j) * height_col + h_out) * width_col + w_out;
    const T offset_h = data_offset_ptr[data_offset_h_ptr];
    const T offset_w = data_offset_ptr[data_offset_w_ptr];
    const T mask = data_mask_ptr[data_mask_hw_ptr];
    const T cur_inv_h_data = h_in + i * dilation_h + offset_h;
    const T cur_inv_w_data = w_in + j * dilation_w + offset_w;

    const T cur_top_grad = data_col[index] * mask;
    const int cur_h = (int)cur_inv_h_data;
    const int cur_w = (int)cur_inv_w_data;
    for (int dy = -2; dy <= 2; dy++) {
      for (int dx = -2; dx <= 2; dx++) {
        if (cur_h + dy >= 0 && cur_h + dy < height && cur_w + dx >= 0 &&
            cur_w + dx < width && abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
            abs(cur_inv_w_data - (cur_w + dx)) < 1) {
          int cur_bottom_grad_pos =
              ((b * channels + c) * height + cur_h + dy) * width + cur_w + dx;
          T weight = dmcn_get_gradient_weight_cpu(cur_inv_h_data,
                                                  cur_inv_w_data, cur_h + dy,
                                                  cur_w + dx, height, width);
          *(grad_im + cur_bottom_grad_pos) += weight * cur_top_grad;
        }
      }
    }
  }
}

template <typename T>
void modulated_deformable_col2im_coord_cpu_kernel(
    const int n, const T *data_col, const T *data_im, const T *data_offset,
    const T *data_mask, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int channel_per_deformable_group,
    const int batch_size, const int offset_channels, const int deformable_group,
    const int height_col, const int width_col, T *grad_offset, T *grad_mask) {
  for (int index = 0; index < n; index++) {
    T val = 0, mval = 0;
    int w = index % width_col;
    int h = (index / width_col) % height_col;
    int c = (index / width_col / height_col) % offset_channels;
    int b = (index / width_col / height_col) / offset_channels;
    // compute the start and end of the output

    const int deformable_group_index = c / (2 * kernel_h * kernel_w);
    const int col_step = kernel_h * kernel_w;
    int cnt = 0;
    const T *data_col_ptr = data_col + deformable_group_index *
                                           channel_per_deformable_group *
                                           batch_size * width_col * height_col;
    const T *data_im_ptr =
        data_im + (b * deformable_group + deformable_group_index) *
                      channel_per_deformable_group / kernel_h / kernel_w *
                      height * width;
    const T *data_offset_ptr =
        data_offset + (b * deformable_group + deformable_group_index) * 2 *
                          kernel_h * kernel_w * height_col * width_col;
    const T *data_mask_ptr =
        data_mask + (b * deformable_group + deformable_group_index) * kernel_h *
                        kernel_w * height_col * width_col;

    const int offset_c = c - deformable_group_index * 2 * kernel_h * kernel_w;

    for (int col_c = (offset_c / 2); col_c < channel_per_deformable_group;
         col_c += col_step) {
      const int col_pos =
          (((col_c * batch_size + b) * height_col) + h) * width_col + w;
      const int bp_dir = offset_c % 2;

      int j = (col_pos / width_col / height_col / batch_size) % kernel_w;
      int i =
          (col_pos / width_col / height_col / batch_size / kernel_w) % kernel_h;
      int w_out = col_pos % width_col;
      int h_out = (col_pos / width_col) % height_col;
      int w_in = w_out * stride_w - pad_w;
      int h_in = h_out * stride_h - pad_h;
      const int data_offset_h_ptr =
          (((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out);
      const int data_offset_w_ptr =
          (((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col +
           w_out);
      const int data_mask_hw_ptr =
          (((i * kernel_w + j) * height_col + h_out) * width_col + w_out);
      const T offset_h = data_offset_ptr[data_offset_h_ptr];
      const T offset_w = data_offset_ptr[data_offset_w_ptr];
      const T mask = data_mask_ptr[data_mask_hw_ptr];
      T inv_h = h_in + i * dilation_h + offset_h;
      T inv_w = w_in + j * dilation_w + offset_w;
      if (inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width)
        inv_h = inv_w = -2;
      else
        mval += data_col_ptr[col_pos] *
                dmcn_im2col_bilinear_cpu(data_im_ptr + cnt * height * width,
                                         width, height, width, inv_h, inv_w);
      const T weight = dmcn_get_coordinate_weight_cpu(
          inv_h, inv_w, height, width, data_im_ptr + cnt * height * width,
          width, bp_dir);
      val += weight * data_col_ptr[col_pos] * mask;
      cnt += 1;
    }
    // KERNEL_ASSIGN(grad_offset[index], offset_req, val);
    grad_offset[index] = val;
    if (offset_c % 2 == 0)
      // KERNEL_ASSIGN(grad_mask[(((b * deformable_group +
      // deformable_group_index) * kernel_h * kernel_w + offset_c / 2) *
      // height_col + h) * width_col + w], mask_req, mval);
      grad_mask[(((b * deformable_group + deformable_group_index) * kernel_h *
                      kernel_w +
                  offset_c / 2) *
                     height_col +
                 h) *
                    width_col +
                w] = mval;
  }
}

void modulated_deformable_im2col_cpu(
    const Tensor data_im, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor data_col) {
  // num_axes should be smaller than block size
  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * batch_size * height_col * width_col;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_im.scalar_type(), "modulated_deformable_im2col_cpu", ([&] {
        const scalar_t *data_im_ = data_im.data_ptr<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data_ptr<scalar_t>();
        const scalar_t *data_mask_ = data_mask.data_ptr<scalar_t>();
        scalar_t *data_col_ = data_col.data_ptr<scalar_t>();

        modulated_deformable_im2col_cpu_kernel(
            num_kernels, data_im_, data_offset_, data_mask_, height_im,
            width_im, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
            dilation_h, dilation_w, channel_per_deformable_group, batch_size,
            channels, deformable_group, height_col, width_col, data_col_);
      }));
}

void modulated_deformable_col2im_cpu(
    const Tensor data_col, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor grad_im) {
  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels =
      channels * kernel_h * kernel_w * batch_size * height_col * width_col;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.scalar_type(), "modulated_deformable_col2im_cpu", ([&] {
        const scalar_t *data_col_ = data_col.data_ptr<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data_ptr<scalar_t>();
        const scalar_t *data_mask_ = data_mask.data_ptr<scalar_t>();
        scalar_t *grad_im_ = grad_im.data_ptr<scalar_t>();

        modulated_deformable_col2im_cpu_kernel(
            num_kernels, data_col_, data_offset_, data_mask_, channels,
            height_im, width_im, kernel_h, kernel_w, pad_h, pad_w, stride_h,
            stride_w, dilation_h, dilation_w, channel_per_deformable_group,
            batch_size, deformable_group, height_col, width_col, grad_im_);
      }));
}

void modulated_deformable_col2im_coord_cpu(
    const Tensor data_col, const Tensor data_im, const Tensor data_offset,
    const Tensor data_mask, const int batch_size, const int channels,
    const int height_im, const int width_im, const int height_col,
    const int width_col, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int deformable_group,
    Tensor grad_offset, Tensor grad_mask) {
  const int num_kernels = batch_size * height_col * width_col * 2 * kernel_h *
                          kernel_w * deformable_group;
  const int channel_per_deformable_group =
      channels * kernel_h * kernel_w / deformable_group;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.scalar_type(), "modulated_deformable_col2im_coord_cpu", ([&] {
        const scalar_t *data_col_ = data_col.data_ptr<scalar_t>();
        const scalar_t *data_im_ = data_im.data_ptr<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data_ptr<scalar_t>();
        const scalar_t *data_mask_ = data_mask.data_ptr<scalar_t>();
        scalar_t *grad_offset_ = grad_offset.data_ptr<scalar_t>();
        scalar_t *grad_mask_ = grad_mask.data_ptr<scalar_t>();

        modulated_deformable_col2im_coord_cpu_kernel(
            num_kernels, data_col_, data_im_, data_offset_, data_mask_,
            channels, height_im, width_im, kernel_h, kernel_w, pad_h, pad_w,
            stride_h, stride_w, dilation_h, dilation_w,
            channel_per_deformable_group, batch_size,
            2 * kernel_h * kernel_w * deformable_group, deformable_group,
            height_col, width_col, grad_offset_, grad_mask_);
      }));
}

void modulated_deformable_im2col_impl(
    const Tensor data_im, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor data_col);

void modulated_deformable_col2im_impl(
    const Tensor data_col, const Tensor data_offset, const Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deformable_group, Tensor grad_im);

void modulated_deformable_col2im_coord_impl(
    const Tensor data_col, const Tensor data_im, const Tensor data_offset,
    const Tensor data_mask, const int batch_size, const int channels,
    const int height_im, const int width_im, const int height_col,
    const int width_col, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int deformable_group,
    Tensor grad_offset, Tensor grad_mask);

REGISTER_DEVICE_IMPL(modulated_deformable_im2col_impl, CPU,
                     modulated_deformable_im2col_cpu);
REGISTER_DEVICE_IMPL(modulated_deformable_col2im_impl, CPU,
                     modulated_deformable_col2im_cpu);
REGISTER_DEVICE_IMPL(modulated_deformable_col2im_coord_impl, CPU,
                     modulated_deformable_col2im_coord_cpu);
