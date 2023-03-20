// Modified from
// https://github.com/facebookresearch/detectron2/tree/master/detectron2/layers/csrc/ROIAlignRotated
// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#include "roi_align_rotated.h"

#include "../ort_mmcv_utils.h"

struct PreCalc {
  int pos1;
  int pos2;
  int pos3;
  int pos4;
  float w1;
  float w2;
  float w3;
  float w4;
};

void pre_calc_for_bilinear_interpolate(
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int iy_upper, const int ix_upper,
    float roi_start_h, float roi_start_w, float bin_size_h, float bin_size_w,
    int roi_bin_grid_h, int roi_bin_grid_w, float roi_center_h,
    float roi_center_w, float cos_theta, float sin_theta,
    std::vector<PreCalc> &pre_calc) {
  int pre_calc_index = 0;
  for (int ph = 0; ph < pooled_height; ph++) {
    for (int pw = 0; pw < pooled_width; pw++) {
      for (int iy = 0; iy < iy_upper; iy++) {
        const float yy =
            roi_start_h + ph * bin_size_h +
            static_cast<float>(iy + .5f) * bin_size_h /
                static_cast<float>(roi_bin_grid_h);  // e.g., 0.5, 1.5
        for (int ix = 0; ix < ix_upper; ix++) {
          const float xx = roi_start_w + pw * bin_size_w +
                           static_cast<float>(ix + .5f) * bin_size_w /
                               static_cast<float>(roi_bin_grid_w);

          // Rotate by theta around the center and translate
          // In image space, (y, x) is the order for Right Handed System,
          // and this is essentially multiplying the point by a rotation matrix
          // to rotate it counterclockwise through angle theta.
          float y = yy * cos_theta - xx * sin_theta + roi_center_h;
          float x = yy * sin_theta + xx * cos_theta + roi_center_w;
          // deal with: inverse elements are out of feature map boundary
          if (y < -1.0 || y > height || x < -1.0 || x > width) {
            // empty
            PreCalc pc;
            pc.pos1 = 0;
            pc.pos2 = 0;
            pc.pos3 = 0;
            pc.pos4 = 0;
            pc.w1 = 0;
            pc.w2 = 0;
            pc.w3 = 0;
            pc.w4 = 0;
            pre_calc[pre_calc_index] = pc;
            pre_calc_index += 1;
            continue;
          }

          if (y < 0) {
            y = 0;
          }
          if (x < 0) {
            x = 0;
          }

          int y_low = (int)y;
          int x_low = (int)x;
          int y_high;
          int x_high;

          if (y_low >= height - 1) {
            y_high = y_low = height - 1;
            y = (float)y_low;
          } else {
            y_high = y_low + 1;
          }

          if (x_low >= width - 1) {
            x_high = x_low = width - 1;
            x = (float)x_low;
          } else {
            x_high = x_low + 1;
          }

          float ly = y - y_low;
          float lx = x - x_low;
          float hy = 1. - ly, hx = 1. - lx;
          float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

          // save weights and indices
          PreCalc pc;
          pc.pos1 = y_low * width + x_low;
          pc.pos2 = y_low * width + x_high;
          pc.pos3 = y_high * width + x_low;
          pc.pos4 = y_high * width + x_high;
          pc.w1 = w1;
          pc.w2 = w2;
          pc.w3 = w3;
          pc.w4 = w4;
          pre_calc[pre_calc_index] = pc;

          pre_calc_index += 1;
        }
      }
    }
  }
}

void ROIAlignRotatedForwardCPU(const int nthreads, const float *input,
                               const float *rois, float *output,
                               const float &spatial_scale, const int aligned,
                               const int clockwise, const int channels,
                               const int height, const int width,
                               const int pooled_height, const int pooled_width,
                               const int sampling_ratio) {
  int n_rois = nthreads / channels / pooled_width / pooled_height;
  // (n, c, ph, pw) is an element in the pooled output
  // can be parallelized using omp
  // #pragma omp parallel for num_threads(32)
  for (int n = 0; n < n_rois; n++) {
    int index_n = n * channels * pooled_width * pooled_height;

    const float *current_roi = rois + n * 6;
    int roi_batch_ind = current_roi[0];

    // Do not use rounding; this implementation detail is critical
    float offset = aligned ? (float)0.5 : (float)0.0;
    float roi_center_w = current_roi[1] * spatial_scale - offset;
    float roi_center_h = current_roi[2] * spatial_scale - offset;
    float roi_width = current_roi[3] * spatial_scale;
    float roi_height = current_roi[4] * spatial_scale;
    // float theta = current_roi[5] * M_PI / 180.0;
    float theta = current_roi[5];  // Radian angle by default
    if (clockwise) {
      theta = -theta;
    }
    float cos_theta = cos(theta);
    float sin_theta = sin(theta);
    if (!aligned) {  // for backward-compatibility only
      roi_width = std::max(roi_width, (float)1.);
      roi_height = std::max(roi_height, (float)1.);
    }

    float bin_size_h =
        static_cast<float>(roi_height) / static_cast<float>(pooled_height);
    float bin_size_w =
        static_cast<float>(roi_width) / static_cast<float>(pooled_width);

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
                             ? sampling_ratio
                             : ceil(roi_height / pooled_height);  // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const float count =
        std::max(roi_bin_grid_h * roi_bin_grid_w, 1);  // e.g. = 4

    // we want to precalculate indices and weights shared by all channels,
    // this is the key point of optimization
    std::vector<PreCalc> pre_calc(roi_bin_grid_h * roi_bin_grid_w *
                                  pooled_width * pooled_height);

    // roi_start_h and roi_start_w are computed wrt the center of RoI (x, y).
    // Appropriate translation needs to be applied after.
    float roi_start_h = -roi_height / 2.0;
    float roi_start_w = -roi_width / 2.0;

    pre_calc_for_bilinear_interpolate(
        height, width, pooled_height, pooled_width, roi_bin_grid_h,
        roi_bin_grid_w, roi_start_h, roi_start_w, bin_size_h, bin_size_w,
        roi_bin_grid_h, roi_bin_grid_w, roi_center_h, roi_center_w, cos_theta,
        sin_theta, pre_calc);

    for (int c = 0; c < channels; c++) {
      int index_n_c = index_n + c * pooled_width * pooled_height;
      const float *offset_input =
          input + (roi_batch_ind * channels + c) * height * width;
      int pre_calc_index = 0;

      for (int ph = 0; ph < pooled_height; ph++) {
        for (int pw = 0; pw < pooled_width; pw++) {
          int index = index_n_c + ph * pooled_width + pw;

          float output_val = 0.;
          for (int iy = 0; iy < roi_bin_grid_h; iy++) {
            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
              PreCalc pc = pre_calc[pre_calc_index];
              output_val += pc.w1 * offset_input[pc.pos1] +
                            pc.w2 * offset_input[pc.pos2] +
                            pc.w3 * offset_input[pc.pos3] +
                            pc.w4 * offset_input[pc.pos4];

              pre_calc_index += 1;
            }
          }
          output_val /= count;

          output[index] = output_val;
        }  // for pw
      }    // for ph
    }      // for c
  }        // for n
}

void MMCVRoIAlignRotatedKernel::Compute(OrtKernelContext *context) {
  // Setup inputs
  const OrtValue *input_X = ort_.KernelContext_GetInput(context, 0);
  const float *X_data =
      reinterpret_cast<const float *>(ort_.GetTensorData<float>(input_X));
  const OrtValue *input_rois = ort_.KernelContext_GetInput(context, 1);
  const float *rois = reinterpret_cast<const float *>(
      ort_.GetTensorData<const float *>(input_rois));

  // Setup output
  OrtTensorDimensions out_dimensions(ort_, input_X);
  OrtTensorDimensions roi_dimensions(ort_, input_rois);

  int batch_size = out_dimensions.data()[0];
  int input_channels = out_dimensions.data()[1];
  int input_height = out_dimensions.data()[2];
  int input_width = out_dimensions.data()[3];

  out_dimensions.data()[0] = roi_dimensions.data()[0];
  out_dimensions.data()[2] = aligned_height_;
  out_dimensions.data()[3] = aligned_width_;

  OrtValue *output = ort_.KernelContext_GetOutput(
      context, 0, out_dimensions.data(), out_dimensions.size());
  float *out = ort_.GetTensorMutableData<float>(output);
  OrtTensorTypeAndShapeInfo *output_info = ort_.GetTensorTypeAndShape(output);
  ort_.ReleaseTensorTypeAndShapeInfo(output_info);

  // TODO: forward here
  int output_size = out_dimensions.data()[0];
  for (auto i = 1; i < out_dimensions.size(); ++i) {
    output_size *= out_dimensions.data()[i];
  }
  ROIAlignRotatedForwardCPU(output_size, X_data, rois, out, spatial_scale_,
                            aligned_, clockwise_, input_channels, input_height,
                            input_width, aligned_height_, aligned_width_,
                            sampling_ratio_);
}
