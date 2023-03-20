/*************************************************************************
 * Copyright (C) 2022 Cambricon.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#ifndef CARAFE_UTILS_HPP_
#define CARAFE_UTILS_HPP_

#define NRAM_ALIGN_SIZE 64

struct CarafeForwardParam {
  int N;   // batch size
  int Hi;  // input height
  int Wi;  // input width
  int Ci;  // input channels
  int Ho;  // output height
  int Wo;  // output width
  int Cg;  // channels per group

  int kernel_size;       // kernel_size
  int group_size;        // group_size
  int scale_factor;      // scale_factor
  int kernel_size_half;  // kernel half size (K-1)/2
  int kernel_size_sq;    // square of kernel size

  int dtype_size;  // size of tensor data type

  // Host arrays' geometry
  int input_stride_g;
  int input_stride_w;
  int input_stride_h;
  int input_stride_n;
  int input_size;
  int mask_stride_kh;
  int mask_stride_g;
  int mask_stride_w;
  int mask_stride_h;
  int mask_stride_n;
  int mask_size;
  int output_stride_g;
  int output_stride_w;
  int output_stride_h;
  int output_stride_n;
  int output_size;

  // NRAM arrays' geometry
  int input_nram_stride_g;
  int input_nram_stride_w;
  int input_nram_stride_h;
  int input_nram_size;
  int mask_nram_stride_kh;
  int mask_nram_stride_g;
  int mask_nram_stride_w;
  int mask_nram_stride_h;
  int mask_nram_size;
  int output_nram_stride_g;
  int output_nram_stride_w;
  int output_nram_stride_h;
  int output_nram_size;

  // for address/compute alignment
  int align_size_NRAM;  // for addressing on NRAM
  int align_size_NFU;   // for NFU operation length
  int block_Cg_NFU;     // for bang_mul_const

  int job_num;  // total job number
};

struct CarafeForwardBlockDim {
  int Ho;  // block size of output height
  int Wo;  // block size of output width
  int Kh;  // block size of kernel height
  int Kw;  // block size of kernel width
  int G;   // block size of groups
  int Cg;  // block size of channels within a group
  int Hi;  // block size of input height
  int Wi;  // block size of input width
};

struct CarafeForwardGridDim {
  int Ho;  // number of blocks of output height
  int Wo;
  int Kh;
  int Kw;
  int G;
  int Cg;
};

#endif  // CARAFE_UTILS_HPP_
