/*************************************************************************
 * Copyright (C) 2021 Cambricon.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#ifndef COMMON_MLU_HELPER_HPP_
#define COMMON_MLU_HELPER_HPP_

#define NFU_ALIGN_SIZE 128          // Byte
#define REM_FOR_STACK (128 * 1024)  // 128KB reserved for cncc

#ifdef __BANG_ARCH__
#define MAX_NRAM_SIZE \
  (__MLU_NRAM_SIZE__ * 1024 - REM_FOR_STACK)  // 128KB reserved for cncc
#define MAX_SRAM_SIZE \
  (__MLU_SRAM_SIZE__ * 1024 - REM_FOR_STACK)  // 128KB reserved for cncc
#else
#define MAX_NRAM_SIZE (384 * 1024)   // 384KB,  initialization value
#define MAX_SRAM_SIZE (1920 * 1024)  // 1920KB, initialization value
#endif

#ifndef PAD_UP
#define PAD_UP(x, y) (((x) / (y) + (int)((x) % (y) > 0)) * (y))
#endif

#ifndef PAD_DOWN
#define PAD_DOWN(x, y) (((x) / (y)) * (y))
#endif

#define CEIL_ALIGN(x, y) (((x) + (y)-1) / (y) * (y))

template <typename scalar_t>
__mlu_func__ inline scalar_t min(scalar_t a, scalar_t b) {
  return a < b ? a : b;
}

template <typename scalar_t>
__mlu_func__ inline scalar_t max(scalar_t a, scalar_t b) {
  return a > b ? a : b;
}

/*!
 * @brief loads data from global DRAM to NRAM with 2D pattern.
 *
 * @param[out] dst
 *   Pointer to NRAM that stores dst data.
 * @param[in] src
 *   Pointer to global DRAM that stores src data.
 * @param[in] size
 *   The byte size of segment in the lower dimension.
 * @param[in] dst_str
 *   The data stride in bytes between segments in the lower dimension of dst.
 * @param[in] src_str
 *   The data stride in bytes between segments in the lower dimension of src.
 * @param[in] seg_num
 *   The total count of data segments in the lower dimension.
 */
template <typename T>
__mlu_func__ void loadStr2D(T *dst, T *src, const int size, const int dst_str,
                            const int src_str, const int seg_num) {
  if (dst_str == src_str && size == src_str) {
    __memcpy(dst, src, src_str * seg_num * sizeof(T), GDRAM2NRAM);
  } else if ((size == src_str || src_str <= dst_str) &&
             src_str * sizeof(T) <= 512) {
    // gather data less than 512Bytes to improve IO efficiency
    T *tmp = (T *)dst + (dst_str - src_str) * seg_num;
    __memcpy(tmp, src, (src_str * (seg_num - 1) + size) * sizeof(T),
             GDRAM2NRAM);
    if (dst_str != src_str) {
      __memcpy(dst, tmp, size * sizeof(T), NRAM2NRAM, dst_str * sizeof(T),
               src_str * sizeof(T), seg_num - 1);
    }
  } else {
    __memcpy(dst, src, size * sizeof(T), GDRAM2NRAM, dst_str * sizeof(T),
             src_str * sizeof(T), seg_num - 1);
  }
}

/*!
 * @brief loads data from global DRAM to NRAM with 3D pattern.
 *
 * @param[out] dst
 *   Pointer to NRAM that stores dst data.
 * @param[in] src
 *   Pointer to global DRAM that stores src data.
 * @param[in] size
 *   The byte size of segment in the lowest dimension.
 * @param[in] seg_num_in
 *   The total count of data segments in the lowest dimension.
 * @param[in] seg_num_out
 *   The total count of data segments in the middle dimension.
 * @param[in] dst_str_in
 *   The data stride in bytes between segments in the lowest dimension of dst.
 * @param[in] dst_str_out
 *   The data stride in bytes between segments in the middle dimension of dst.
 * @param[in] src_str_in
 *   The data stride in bytes between segments in the lowest dimension of src.
 * @param[in] src_str_out
 *   The data stride in bytes between segments in the middle dimension of src.
 */
template <typename T>
__mlu_func__ void loadStr3D(T *dst, T *src, const int size,
                            const int seg_num_in, const int seg_num_out,
                            const int dst_str_in, const int dst_str_out,
                            const int src_str_in, const int src_str_out) {
  T *tmp_dst = dst;
  T *tmp_src = src;

  for (int i = 0; i < seg_num_out; ++i) {
    loadStr2D(tmp_dst, tmp_src, size, dst_str_in, src_str_in, seg_num_in);
    tmp_src += src_str_out;
    tmp_dst += dst_str_out;
  }
}

/*!
 * @brief stores data from NRAM to global DRAM with 2D pattern.
 *
 * @param[out] dst
 *   Pointer to global DRAM that stores dst data.
 * @param[in] src
 *   Pointer to NRAM that stores src data.
 * @param[in] size
 *   The byte size of segment in the lower dimension.
 * @param[in] dst_str
 *   The data stride in bytes between segments in the lower dimension of dst.
 * @param[in] src_str
 *   The data stride in bytes between segments in the lower dimension of src.
 * @param[in] seg_num
 *   The total count of data segments in the lower dimension.
 */
template <typename T>
__mlu_func__ void storeStr2D(T *dst, T *src, const int size, const int seg_num,
                             const int dst_str, const int src_str) {
  if ((size == dst_str && dst_str <= src_str) && dst_str * sizeof(T) <= 512) {
    // gather data less than 512Bytes to improve IO efficiency
    if (dst_str != src_str) {
      __memcpy(src, src, size * sizeof(T), NRAM2NRAM, dst_str * sizeof(T),
               src_str * sizeof(T), seg_num - 1);
    }
    __memcpy(dst, src, size * seg_num * sizeof(T), NRAM2GDRAM);
  } else {
    __memcpy(dst, src, size * sizeof(T), NRAM2GDRAM, dst_str * sizeof(T),
             src_str * sizeof(T), seg_num - 1);
  }
}

/*!
 * @brief stores data from NRAM to global DRAM with 3D pattern.
 *
 * @param[out] dst
 *   Pointer to global DRAM that stores dst data.
 * @param[in] src
 *   Pointer to NRAM that stores src data.
 * @param[in] size
 *   The byte size of segment in the lowest dimension.
 * @param[in] seg_num_in
 *   The total count of data segments in the lowest dimension.
 * @param[in] seg_num_out
 *   The total count of data segments in the middle dimension.
 * @param[in] dst_str_in
 *   The data stride in bytes between segments in the lowest dimension of dst.
 * @param[in] dst_str_out
 *   The data stride in bytes between segments in the middle dimension of dst.
 * @param[in] src_str_in
 *   The data stride in bytes between segments in the lowest dimension of src.
 * @param[in] src_str_out
 *   The data stride in bytes between segments in the middle dimension of src.
 */
template <typename T>
__mlu_func__ void storeStr3D(T *dst, T *src, const int size,
                             const int seg_num_in, const int seg_num_out,
                             const int dst_str_in, const int dst_str_out,
                             const int src_str_in, const int src_str_out) {
  T *tmp_dst = dst;
  T *tmp_src = src;
  for (int i = 0; i < seg_num_out; ++i) {
    storeStr2D(tmp_dst, tmp_src, size, seg_num_in, dst_str_in, src_str_in);
    tmp_src += src_str_out;
    tmp_dst += dst_str_out;
  }
}

/*!
 * @brief Converts int32 to float32 data type.
 *
 * @param[out] dst
 *   Pointer to NRAM that stores int32 type data.
 * @param[in,out] dst_addition
 *   Pointer to NRAM as the workspace of dst, which has the same size as dst.
 *   It allows empty pointer on MLU300 series.
 * @param[in] src
 *   Pointer to NRAM that stores float32 type data.
 * @param[in,out] src_addition
 *   Pointer to NRAM as the workspace of src, which has a size of 128 Bytes.
 *   It allows empty pointer on MLU300 series.
 * @param[in] src_count
 *   The count of elements in src.
 */
__mlu_func__ void convertInt2Float(float *dst, float *dst_addition, int *src,
                                   float *src_addition, const int src_count) {
#if __BANG_ARCH__ >= 300
  __bang_int2float((float *)dst, (int32_t *)src, src_count, 0);
#else
  // get sign bit
  const float move_23bit = 8388608.0;
  // 0x80000000 = 1,000000000,0000000000000000000000000000
  __bang_write_value((unsigned *)src_addition, NFU_ALIGN_SIZE / sizeof(float),
                     0x80000000);
  __bang_cycle_band((char *)dst_addition, (char *)src, (char *)src_addition,
                    src_count * sizeof(float), NFU_ALIGN_SIZE);
  // get 1 or 0 from sign bit
  // judg is Odd
  __bang_write_value((unsigned *)src_addition, NFU_ALIGN_SIZE / sizeof(float),
                     0x00000001);
  __bang_cycle_bor((char *)dst_addition, (char *)dst_addition,
                   (char *)src_addition, src_count * sizeof(float),
                   NFU_ALIGN_SIZE);
  __bang_write_value((unsigned *)src_addition, NFU_ALIGN_SIZE / sizeof(float),
                     0x80000001);
  __bang_cycle_eq(dst_addition, dst_addition, src_addition, src_count,
                  NFU_ALIGN_SIZE / sizeof(float));
  // minus xor, positive num invariant
  __bang_write_value((unsigned *)src_addition, NFU_ALIGN_SIZE / sizeof(float),
                     0xffffffff);
  __bang_cycle_mul(dst, dst_addition, src_addition, src_count,
                   NFU_ALIGN_SIZE / sizeof(float));
  __bang_bxor((char *)dst, (char *)src, (char *)dst, src_count * sizeof(float));
  // convert int32 to float32
  __bang_write_value((unsigned *)src_addition, NFU_ALIGN_SIZE / sizeof(float),
                     0x7fffff);
  __bang_cycle_band((char *)dst, (char *)dst, (char *)src_addition,
                    src_count * sizeof(float), NFU_ALIGN_SIZE);
  __bang_write_value((unsigned *)src_addition, NFU_ALIGN_SIZE / sizeof(float),
                     0x4b000000);
  __bang_cycle_bor((char *)dst, (char *)dst, (char *)src_addition,
                   src_count * sizeof(float), NFU_ALIGN_SIZE);
  __bang_sub_scalar(dst, dst, move_23bit, src_count);
  // add one
  __bang_add(dst, dst, dst_addition, src_count);
  // set sign for float32
  __bang_write_value((unsigned *)src_addition, NFU_ALIGN_SIZE / sizeof(float),
                     0xffffffff);
  __bang_cycle_mul(dst_addition, dst_addition, src_addition, src_count,
                   NFU_ALIGN_SIZE / sizeof(float));

  __bang_write_value((unsigned *)src_addition, NFU_ALIGN_SIZE / sizeof(float),
                     0x00000001);
  __bang_cycle_add(dst_addition, dst_addition, src_addition, src_count,
                   NFU_ALIGN_SIZE / sizeof(float));

  __bang_write_value((unsigned *)src_addition, NFU_ALIGN_SIZE / sizeof(float),
                     0x80000000);
  __bang_cycle_band((char *)dst_addition, (char *)dst_addition,
                    (char *)src_addition, src_count * 4, 128);
  __bang_bor((char *)dst, (char *)dst, (char *)dst_addition, src_count * 4);
#endif  // __BANG_ARCH__ >= 300
}

/*!
 * @brief Converts float32 to int32 data type with to_zero round mode.
 *
 * @param[out] dst
 *   Pointer to NRAM that stores float32 type data.
 * @param[in,out] dst_addition
 *   Pointer to NRAM as the workspace of dst, which has the same size as dst.
 *   It allows empty pointer on MLU300 series.
 * @param[in] src
 *   Pointer to NRAM that stores int32 type data.
 * @param[in,out] src_addition
 *   Pointer to NRAM as the workspace of src, which has a size of 128 Bytes.
 *   It allows empty pointer on MLU300 series.
 * @param[in] src_count
 *   The count of elements in src.
 */
__mlu_func__ void convertFloat2Int(int *dst, float *dst_addition, float *src,
                                   float *src_addition, const int src_count) {
#if __BANG_ARCH__ >= 300
  __bang_float2int_tz((int32_t *)dst, (float *)src, src_count, 0);
#else
  // sign ===> src_addition
  // dst=-1.0 : when src[i] is a negative number
  // dst=+1.0 : when src[i] is a positive number
  const int floatDchar = sizeof(float) / sizeof(char);
  __bang_active_sign((float *)dst, src, src_count);
  // dst_addition = abs(src)
  __bang_mul(dst_addition, src, (float *)dst, src_count);
  // if dst_addition < 1.0 , then src_addition + 1, to fix add error.
  __bang_write_value((float *)src_addition, NFU_ALIGN_SIZE / sizeof(float),
                     1.0f);
  __bang_cycle_lt(dst_addition, dst_addition, (float *)src_addition, src_count,
                  NFU_ALIGN_SIZE / sizeof(float));
  __bang_add_tz((float *)dst, (float *)dst, (float *)dst_addition, src_count);
  __bang_write_value((unsigned *)src_addition, NFU_ALIGN_SIZE / sizeof(float),
                     0xbf800000);
  // set negative flag -1.0 = 0xbf80000
  __bang_cycle_eq(
      (float *)dst, (float *)dst, (float *)src_addition, src_count,
      NFU_ALIGN_SIZE / sizeof(float));  //  to mark all src in [x<-1.0]
  __bang_active_abs(dst_addition, src, src_count);
  __bang_write_value((float *)src_addition, NFU_ALIGN_SIZE / sizeof(float),
                     8388608.0f);
  // mask shift move 23
  __bang_cycle_add_tz(
      dst_addition, dst_addition, src_addition, src_count,
      NFU_ALIGN_SIZE / sizeof(float));  // right shift move 23bit
  // two`s complement for negatibe
  // dst=1.0 , when src <-1.0
  // dst=0.0 , when src >=-1.0
  __bang_sub(dst_addition, dst_addition, (float *)dst, src_count);
  // to fix max value
  // 0 1001 0110 111 1111 1111 1111 1111 1111 <=> 0xcb7fffff <=> 16777215.0,
  // means max value.
  __bang_mul_scalar((float *)dst, (float *)dst, 16777215.0, src_count);
  __bang_bxor((char *)dst_addition, (char *)dst_addition, (char *)dst,
              src_count * floatDchar);
  // get low 23bit
  __bang_write_value((unsigned *)src_addition, NFU_ALIGN_SIZE / sizeof(float),
                     (unsigned)0x007fffff);
  // mask low 23bit is 1
  __bang_cycle_band((char *)dst_addition, (char *)dst_addition,
                    (char *)src_addition, src_count * floatDchar,
                    NFU_ALIGN_SIZE / sizeof(char));
  // set 9 high bit ===> dst
  // -2.0 <=> 0xc0000000 <=> 1100 0000 0000 0000 0000 0000 0000 0000
  //  1.0 <=> 0x3f800000 <=> 0011 1111 1000 0000 0000 0000 0000 0000
  __bang_write_value(src_addition, NFU_ALIGN_SIZE / sizeof(float), 0x3f800000);
  __bang_cycle_and((float *)dst, (float *)dst, src_addition, src_count,
                   NFU_ALIGN_SIZE / sizeof(float));
  // src or dst_addition
  __bang_bor((char *)dst_addition, (char *)dst, (char *)dst_addition,
             src_count * floatDchar);
  __bang_mul_scalar((float *)dst, (float *)dst, -2.0, src_count);
  __bang_bor((char *)dst, (char *)dst, (char *)dst_addition,
             src_count * floatDchar);
#endif  // __BANG_ARCH__ >= 300
}

/*!
 * @brief Converts float32 to half data type,
 * the rounding mode on MLU200 is rd, on MLU300 is rn.
 *
 * @param[out] dst
 *   Pointer to NRAM that stores half type data.
 * @param[in] src
 *   Pointer to NRAM that stores float32 type data.
 * @param[in] src_count
 *   The count of elements in src.
 */
__mlu_func__ inline void convertFloat2half(half *dst, float *src,
                                           int src_count) {
#if __BANG_ARCH__ >= 300
  __bang_float2half_rn(dst, src, src_count);
#else
  __bang_float2half_rd(dst, src, src_count);
#endif
}

/*!
 * @brief recursiveSumPool.
 * @param[in,out] dst
 *     Pointer to NRAM that stores the input and output data.
 * @param[in] low_dim
 *     Which is the number of low dim.
 * @param[in] high_dim
 *     Which is the number of high dim.
 * @param[in] kernel_limit
 *     Which is the high_dim of sumpool per time.
 ******************************************************************************/
template <typename T>
__mlu_func__ void recursiveSumPool(T *dst, int low_dim, int high_dim,
                                   int kernel_limit) {
  for (; high_dim > 1;) {
    int repeat_s = high_dim / kernel_limit;
    int remain_s = high_dim % kernel_limit;

    if (remain_s) {
      __bang_sumpool((T *)dst, (T *)dst, low_dim, 1, remain_s, 1, remain_s, 1,
                     1);
    }
    if (repeat_s) {
      __bang_sumpool((T *)dst + (remain_s > 0 ? low_dim : 0),
                     (T *)dst + remain_s * low_dim, low_dim,
                     kernel_limit * repeat_s, 1, kernel_limit, 1, 1,
                     kernel_limit);
    }
    high_dim = repeat_s + (bool)remain_s;
  }
  return;
}

#endif  // COMMON_MLU_HELPER_HPP_
