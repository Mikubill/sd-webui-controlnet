// Copyright (c) OpenMMLab. All rights reserved
#include "corner_pool.h"

#include "../ort_mmcv_utils.h"

void TopPoolForwardCPU(const float *input, float *output, const int batch_size,
                       const int channels, const int height, const int width) {
  for (int n = 0; n < batch_size; n++) {
    int index_n = n * channels * width * height;
    for (int c = 0; c < channels; c++) {
      int index_n_c = index_n + c * width * height;
      for (int w = 0; w < width; w++) {
        // directly copy the most bottom value from input to output
        output[index_n_c + (height - 1) * width + w] =
            input[index_n_c + (height - 1) * width + w];
        // do top_pool
        for (int h = height - 2; h >= 0; h--) {
          output[index_n_c + h * width + w] =
              std::max(output[index_n_c + (h + 1) * width + w],
                       input[index_n_c + h * width + w]);
        }  // for h
      }    // for w
    }      // for c
  }        // for n
}

void BottomPoolForwardCPU(const float *input, float *output,
                          const int batch_size, const int channels,
                          const int height, const int width) {
  for (int n = 0; n < batch_size; n++) {
    int index_n = n * channels * width * height;
    for (int c = 0; c < channels; c++) {
      int index_n_c = index_n + c * width * height;
      for (int w = 0; w < width; w++) {
        // directly copy the most top value from input to output
        output[index_n_c + w] = input[index_n_c + w];
        // do top_pool
        for (int h = 1; h < height; h++) {
          output[index_n_c + h * width + w] =
              std::max(output[index_n_c + (h - 1) * width + w],
                       input[index_n_c + h * width + w]);
        }  // for h
      }    // for w
    }      // for c
  }        // for n
}

void LeftPoolForwardCPU(const float *input, float *output, const int batch_size,
                        const int channels, const int height, const int width) {
  for (int n = 0; n < batch_size; n++) {
    int index_n = n * channels * width * height;
    for (int c = 0; c < channels; c++) {
      int index_n_c = index_n + c * width * height;
      for (int h = 0; h < height; h++) {
        // directly copy the most right value from input to output
        output[index_n_c + h * width + width - 1] =
            input[index_n_c + h * width + width - 1];
        // do left_pool
        for (int w = width - 2; w >= 0; w--) {
          output[index_n_c + h * width + w] =
              std::max(output[index_n_c + h * width + w + 1],
                       input[index_n_c + h * width + w]);
        }  // for w
      }    // for h
    }      // for c
  }        // for n
}

void RightPoolForwardCPU(const float *input, float *output,
                         const int batch_size, const int channels,
                         const int height, const int width) {
  for (int n = 0; n < batch_size; n++) {
    int index_n = n * channels * width * height;
    for (int c = 0; c < channels; c++) {
      int index_n_c = index_n + c * width * height;
      for (int h = 0; h < height; h++) {
        // directly copy the most left value from input to output
        output[index_n_c + h * width] = input[index_n_c + h * width];
        // do right_pool
        for (int w = 1; w < width; w++) {
          output[index_n_c + h * width + w] =
              std::max(output[index_n_c + h * width + w - 1],
                       input[index_n_c + h * width + w]);
        }  // for w
      }    // for h
    }      // for c
  }        // for n
}

void MMCVCornerPoolKernel::Compute(OrtKernelContext *context) {
  const int mode = int(mode_);
  typedef float T;
  const OrtValue *input = ort_.KernelContext_GetInput(context, 0);
  const T *input_data =
      reinterpret_cast<const float *>(ort_.GetTensorData<T>(input));

  // get output memory
  OrtTensorDimensions out_dimensions(ort_, input);
  OrtValue *output = ort_.KernelContext_GetOutput(
      context, 0, out_dimensions.data(), out_dimensions.size());
  T *output_data = ort_.GetTensorMutableData<T>(output);

  // 'top': 0, 'bottom': 1, 'left': 2, 'right':3
  assert(mode == 0 || mode == 1 || mode == 2 || mode == 3);

  // do corner_pool
  int batch_size = out_dimensions.data()[0];
  int input_channels = out_dimensions.data()[1];
  int input_height = out_dimensions.data()[2];
  int input_width = out_dimensions.data()[3];
  if (mode == 0)
    TopPoolForwardCPU(input_data, output_data, batch_size, input_channels,
                      input_height, input_width);
  else if (mode == 1)
    BottomPoolForwardCPU(input_data, output_data, batch_size, input_channels,
                         input_height, input_width);
  else if (mode == 2)
    LeftPoolForwardCPU(input_data, output_data, batch_size, input_channels,
                       input_height, input_width);
  else
    RightPoolForwardCPU(input_data, output_data, batch_size, input_channels,
                        input_height, input_width);
}
