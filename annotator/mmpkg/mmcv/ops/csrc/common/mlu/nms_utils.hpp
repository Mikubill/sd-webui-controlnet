/*************************************************************************
 * Copyright (C) [2019-2022] by Cambricon, Inc.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#ifndef NMS_UTILS_HPP_
#define NMS_UTILS_HPP_
#include "common_mlu_helper.hpp"

#define NMS_SIZE (64)
#define NMS_UP(x, y) (x / y + (int)(x % y > 0)) * y
#define NMS_DOWN(x, y) (x / y) * y
#define INFO_NUM (5)  // 5 means x1, x2, y1, y2 and score
#define MEMORY_CORE (0x80)
#define REDUCE_NUM \
  (7)  // score, x1, y1, x2, y2, max_index (reserve 2 num for half-type input)

__mlu_func__ void pvLock() {
#if __BANG_ARCH__ == 270
  if (coreId != MEMORY_CORE) {
    __bang_lock(0, 0);
  }
#endif
}

__mlu_func__ void pvUnlock() {
#if __BANG_ARCH__ == 270
  if (coreId != MEMORY_CORE) {
    __bang_unlock(0, 0);
  }
#endif
}

template <typename T>
static __mlu_func__ void computeReluN(T *nram_dst, T *nram_src, void *nram_tmp,
                                      const int deal_num,
                                      const T threshold = 0) {
  if (threshold < 0) {
    return;
  }
  if (threshold) {
#if __BANG_ARCH__ >= 300
    __bang_relun(nram_dst, nram_src, deal_num, threshold);
#else
    int align_num = NFU_ALIGN_SIZE / sizeof(T);
    T *nram_aux_a = (T *)nram_tmp;
    T *nram_aux_b = nram_aux_a + deal_num;
    T *nram_zero = nram_aux_b + align_num;
    __bang_write_value(nram_aux_b, align_num, threshold);
    __bang_write_zero(nram_zero, align_num);
    __bang_cycle_lt((T *)nram_aux_a, nram_src, (T *)nram_aux_b, deal_num,
                    align_num);
    __bang_mul(nram_dst, nram_src, (T *)nram_aux_a, deal_num);
    __bang_cycle_eq((T *)nram_aux_a, (T *)nram_aux_a, (T *)nram_zero, deal_num,
                    align_num);
    __bang_cycle_mul((T *)nram_aux_a, (T *)nram_aux_a, (T *)nram_aux_b,
                     deal_num, align_num);
    __bang_add(nram_dst, nram_dst, (T *)nram_aux_a, deal_num);
    __bang_cycle_gt((T *)nram_aux_a, nram_dst, (T *)nram_zero, deal_num,
                    align_num);
    __bang_mul(nram_dst, nram_dst, (T *)nram_aux_a, deal_num);
#endif
  } else {
#if __BANG_ARCH__ >= 300
    __bang_relu(nram_dst, nram_src, deal_num);
#else
    __bang_active_relu(nram_dst, nram_src, deal_num);
#endif
  }
}

__mlu_func__ void getComputeParamsBlockOrU1(
    const int input_dwidth, const int input_box_num, const int limit,
    const int core_limit, int &input_offset, int &max_seg_pad, int &repeat,
    int &remain, int &remain_pad, int &max_seg_iou_compute,
    int &repeat_iou_compute, int &remain_iou_compute,
    int &remain_pad_iou_compute) {
  int avg_core = input_box_num / core_limit;
  int rem = input_box_num % core_limit;
  int len_core = avg_core + (coreId < rem ? 1 : 0);
  input_offset = avg_core * coreId + (coreId <= rem ? coreId : rem);
  max_seg_pad = NMS_DOWN(limit, NMS_SIZE);
  repeat = len_core / max_seg_pad;
  remain = len_core % max_seg_pad;
  remain_pad = NMS_UP(remain, NMS_SIZE);

  // if datatype is fp16, we should cvt to fp32 when compute iou
  max_seg_iou_compute = NMS_DOWN(max_seg_pad / (4 / input_dwidth), NMS_SIZE);
  repeat_iou_compute = len_core / max_seg_iou_compute;
  remain_iou_compute = len_core % max_seg_iou_compute;
  remain_pad_iou_compute = NMS_UP(remain_iou_compute, NMS_SIZE);
}

__mlu_func__ void getComputeParamsUx(
    const int input_dwidth, const int input_num_boxes, const int limit,
    int &input_offset, int &max_seg_pad, int &repeat, int &remain,
    int &remain_pad, int &max_seg_iou_compute, int &repeat_iou_compute,
    int &remain_iou_compute, int &remain_pad_iou_compute) {
  // data split
  int avg_cluster = input_num_boxes / clusterDim;
  int rem_cluster = input_num_boxes % clusterDim;
  int len_cluster = avg_cluster + (clusterId < rem_cluster);
  int cluster_offset = avg_cluster * clusterId +
                       (clusterId <= rem_cluster ? clusterId : rem_cluster);

  int avg_core = len_cluster / coreDim;
  int rem_core = len_cluster % coreDim;
  int len_core = avg_core + (coreId < rem_core);
  int core_offset =
      avg_core * coreId + (coreId <= rem_core ? coreId : rem_core);
  input_offset = cluster_offset + core_offset;

  max_seg_pad = NMS_DOWN(limit, NMS_SIZE);

  // core 0 of each cluster calculate the max score index
  int max_index_len_core = avg_cluster + (clusterId < rem_cluster);
  repeat = max_index_len_core / max_seg_pad;
  remain = max_index_len_core % max_seg_pad;
  remain_pad = NMS_UP(remain, NMS_SIZE);
  // if datatype is fp16, we should cvt to fp32 when compute iou
  max_seg_iou_compute =
      NMS_DOWN(max_seg_pad / (sizeof(float) / input_dwidth), NMS_SIZE);
  repeat_iou_compute = len_core / max_seg_iou_compute;
  remain_iou_compute = len_core % max_seg_iou_compute;
  remain_pad_iou_compute = NMS_UP(remain_iou_compute, NMS_SIZE);
}

template <typename IN_DT>
__mlu_func__ void findGlobalMaxBox(IN_DT *max_box, IN_DT *sram,
                                   IN_DT *inter_x1) {
  // copy all partial max to the sram of cluster 0
  if (clusterId != 0) {
    __memcpy(sram + REDUCE_NUM * clusterId, sram, REDUCE_NUM * sizeof(IN_DT),
             SRAM2SRAM, 0);
  }
  __sync_all();

  // reduce between clusters to get the global max box
  if (clusterId == 0) {
    if (coreId == 0) {
      __bang_write_zero(inter_x1, NMS_SIZE);
      __memcpy(inter_x1, sram, sizeof(IN_DT), SRAM2NRAM, sizeof(IN_DT),
               REDUCE_NUM * sizeof(IN_DT), clusterDim - 1);
      __bang_max(max_box, inter_x1, NMS_SIZE);
      int max_cluster = (sizeof(IN_DT) == sizeof(half))
                            ? ((uint16_t *)max_box)[1]
                            : ((uint32_t *)max_box)[1];
      __memcpy(max_box, sram + max_cluster * REDUCE_NUM,
               REDUCE_NUM * sizeof(IN_DT), SRAM2NRAM);
      __memcpy(sram, max_box, REDUCE_NUM * sizeof(IN_DT), NRAM2SRAM);
    }
    __sync_cluster();
    if (coreId == 0x80 && clusterDim > 1) {
      // broadcast global max box to each cluster's sram
      for (int cluster_idx = 1; cluster_idx < clusterDim; ++cluster_idx) {
        __memcpy(sram, sram, REDUCE_NUM * sizeof(IN_DT), SRAM2SRAM,
                 cluster_idx);
      }
    }
    __sync_cluster();
  }
  __sync_all();

  // copy the global max box to max_box
  __memcpy(max_box, sram, REDUCE_NUM * sizeof(IN_DT), SRAM2NRAM);
}

template <typename IN_DT>
__mlu_func__ void findCoreMaxBox(
    IN_DT *input_score_ptr, IN_DT *score, IN_DT *inter_x1, IN_DT *max_box,
    const IN_DT *input_x1_ptr, const IN_DT *input_y1_ptr,
    const IN_DT *input_x2_ptr, const IN_DT *input_y2_ptr,
    const mluMemcpyDirection_t load_dir, const int input_offset,
    const int repeat, const int remain, const int remain_pad,
    const int max_seg_pad, int &max_index) {
  if (coreId != 0x80) {
    for (int i = 0; i <= repeat; i++) {
      if (i == repeat && remain == 0) {
        break;
      }
      int seg_len = 0;  // the length every nms compute
      int cpy_len = 0;  // the length every nms memcpy
      i == repeat ? seg_len = remain_pad : seg_len = max_seg_pad;
      i == repeat ? cpy_len = remain : cpy_len = max_seg_pad;
      /******NMS LOAD START******/
      __bang_write_zero(score, seg_len);
      __memcpy(score, input_score_ptr + input_offset + i * max_seg_pad,
               cpy_len * sizeof(IN_DT), load_dir, cpy_len * sizeof(IN_DT),
               cpy_len * sizeof(IN_DT), 0);

      /******NMS LOAD END******/

      __bang_max(inter_x1, score, seg_len);
      if (inter_x1[0] > max_box[0]) {
        max_box[0] = inter_x1[0];
        if (sizeof(IN_DT) == sizeof(half)) {
          max_index = ((uint16_t *)inter_x1)[1] + input_offset +
                      i * max_seg_pad;  // offset start from head of input_data
        } else if (sizeof(IN_DT) == sizeof(float)) {
          max_index = ((uint32_t *)inter_x1)[1] + input_offset +
                      i * max_seg_pad;  // offset start from head of input_data
        }
      }
    }  // for repeat
    // the max box's x1, y1, x2, y2 on every core
    max_box[1] = input_x1_ptr[max_index];
    max_box[2] = input_y1_ptr[max_index];
    max_box[3] = input_x2_ptr[max_index];
    max_box[4] = input_y2_ptr[max_index];
    ((uint32_t *)(max_box + 5))[0] = max_index;
  }
}

template <typename IN_DT>
__mlu_func__ void findClusterMaxBox(IN_DT *sram, IN_DT *max_box,
                                    IN_DT *inter_x1, IN_DT *input_data_score,
                                    const int core_limit) {
  // find the max with sram
  // copy every core's box info to sram, form: score---x1---y1---x2---y2---
  __memcpy(sram + REDUCE_NUM * coreId, max_box, REDUCE_NUM * sizeof(IN_DT),
           NRAM2SRAM);  // int32_t datatype
  __sync_cluster();

  // copy score from sram to nram and find the max
  __bang_write_zero(inter_x1, 64);
  __memcpy(inter_x1, sram, sizeof(IN_DT), SRAM2NRAM, sizeof(IN_DT),
           REDUCE_NUM * sizeof(IN_DT), coreDim - 1);
  __bang_max(max_box, inter_x1, 64);
  int max_core = sizeof(IN_DT) == sizeof(half) ? ((uint16_t *)max_box)[1]
                                               : ((uint32_t *)max_box)[1];
  // copy the max box to max_box
  __memcpy(max_box, sram + max_core * REDUCE_NUM, REDUCE_NUM * sizeof(IN_DT),
           SRAM2NRAM);
}

/*****************************************************************************/
/*******************************CALCULATE MAX AREA****************************/
/*****************************************************************************/

template <typename IN_DT>
__mlu_func__ void calMaxArea(IN_DT *max_box, const int algo, float offset,
                             float &max_area) {
  if (algo == 0 || offset == 0.0) {
    max_area = ((float)max_box[3] - (float)max_box[1]) *
               ((float)max_box[4] - (float)max_box[2]);
  } else {
    max_area = ((float)max_box[3] - (float)max_box[1] + offset) *
               ((float)max_box[4] - (float)max_box[2] + offset);
  }
}

template <typename IN_DT>
__mlu_func__ void calMaxArea(IN_DT *max_box, const int algo, float offset,
                             float &max_area, float &max_box_x1,
                             float &max_box_y1, float &max_box_x2,
                             float &max_box_y2) {
  // the case of random inf will break the requirement of x1<=x2, y1<=y2
  // so exchange it if it happens.
  max_box_x1 = float(max_box[1]);
  max_box_x2 = float(max_box[3]);
  if (max_box[1] > max_box[3]) {
    max_box_x1 = float(max_box[3]);
    max_box_x2 = float(max_box[1]);
  }
  max_box_y1 = float(max_box[2]);
  max_box_y2 = float(max_box[4]);
  if (max_box[2] > max_box[4]) {
    max_box_y1 = float(max_box[4]);
    max_box_y2 = float(max_box[2]);
  }
  if (algo == 0 || offset == 0.0) {
    max_area = (max_box_x2 - max_box_x1) * (max_box_y2 - max_box_y1);
  } else {
    max_area =
        (max_box_x2 - max_box_x1 + offset) * (max_box_y2 - max_box_y1 + offset);
  }
}

/***********************************************************************/
/*******************************STORE RESULT****************************/
/***********************************************************************/
template <typename IN_DT, typename OUT_DT>
__mlu_func__ void storeResult(IN_DT *max_box, OUT_DT *nram_save,
                              OUT_DT *&output_dram, const int keep,
                              const int nram_save_limit_count,
                              const int max_output_size,
                              const float thresh_score, const int output_mode,
                              int &nram_save_count, uint32_t &output_box_num) {
  /******NMS STORE START******/
  // store to nram
  if (float(max_box[0]) > thresh_score) {
    OUT_DT *save_ptr;
    int save_offset = 0;
    int save_str_num = 0;
    save_ptr = nram_save;
    save_offset = nram_save_count;
    save_str_num = nram_save_limit_count;
    if (clusterId == 0 && coreId == 0) {
      if (output_mode == 0) {  // index1, index2, ...
        save_ptr[save_offset] = ((uint32_t *)(max_box + INFO_NUM))[0];
      } else if (output_mode == 1) {  // score, x1, y1, x2, y2
        __memcpy(save_ptr + save_offset * INFO_NUM, max_box,
                 INFO_NUM * sizeof(IN_DT), NRAM2NRAM, INFO_NUM * sizeof(IN_DT),
                 INFO_NUM * sizeof(IN_DT), 0);
      } else if (output_mode == 2) {  // score---, x1---, y1---, x2---, y2---
        __memcpy(save_ptr + save_offset, max_box, 1 * sizeof(IN_DT), NRAM2NRAM,
                 save_str_num * sizeof(IN_DT), 1 * sizeof(IN_DT), 4);
      }
    }
    nram_save_count++;
    output_box_num++;
  }

  // store to sram/gdram
  if (output_box_num != 0) {
    if ((nram_save_count == nram_save_limit_count) ||
        (float(max_box[0]) <= thresh_score) || keep == max_output_size - 1) {
      if (nram_save_count != 0) {
        if (clusterId == 0 && coreId == 0) {
          if (output_mode == 0) {  // index1, index2, ...
            pvLock();
            __memcpy(output_dram, nram_save, nram_save_count * sizeof(uint32_t),
                     NRAM2GDRAM);
            pvUnlock();
            output_dram += nram_save_count;
          } else if (output_mode == 1) {  // score, x1, y1, x2, y2
            pvLock();
            __memcpy(output_dram, nram_save,
                     nram_save_count * INFO_NUM * sizeof(IN_DT), NRAM2GDRAM);
            pvUnlock();
            output_dram += nram_save_count * INFO_NUM;
          } else if (output_mode ==
                     2) {  // score---, x1---, y1---, x2---, y2---
            pvLock();
            __memcpy(output_dram, nram_save, nram_save_count * sizeof(IN_DT),
                     NRAM2GDRAM, max_output_size * sizeof(IN_DT),
                     nram_save_limit_count * sizeof(IN_DT), 4);
            pvUnlock();
            output_dram += nram_save_count;
          }
          nram_save_count = 0;
        }
      }
    }  // if move data nram->sram/gdram
  }    // if dst
}

template <typename IN_DT, typename OUT_DT>
__mlu_func__ void scoreUpdate(
    IN_DT *input_score_ptr, const mluMemcpyDirection_t load_dir,
    const mluMemcpyDirection_t store_dir, const IN_DT *input_x1_ptr,
    const IN_DT *input_y1_ptr, const IN_DT *input_x2_ptr,
    const IN_DT *input_y2_ptr, IN_DT *x1, IN_DT *y1, IN_DT *x2, IN_DT *y2,
    IN_DT *score, IN_DT *inter_x1, IN_DT *inter_y1, IN_DT *inter_x2,
    IN_DT *inter_y2, IN_DT *max_box, const float max_box_x1,
    const float max_box_y1, const float max_box_x2, const float max_box_y2,
    OUT_DT *nram_save, int repeat_iou_compute, int remain_iou_compute,
    int remain_pad_iou_compute, int max_seg_iou_compute, int max_seg_pad,
    const float thresh_iou, const float div_thresh_iou, const int input_offset,
    const float offset, const float max_area, const int input_num_boxes,
    const int algo) {
  for (int i = 0; i <= repeat_iou_compute; i++) {
    if (i == repeat_iou_compute && remain_iou_compute == 0) {
      break;
    }
    int seg_len = (i == repeat_iou_compute) ? remain_pad_iou_compute
                                            : max_seg_iou_compute;
    int cpy_len =
        (i == repeat_iou_compute) ? remain_iou_compute : max_seg_iou_compute;
    /******NMS LOAD START******/
    int dt_offset = 0;
    if (sizeof(IN_DT) == sizeof(float)) {
      __memcpy(score, input_score_ptr + input_offset + i * max_seg_pad,
               cpy_len * sizeof(IN_DT), load_dir, cpy_len * sizeof(IN_DT),
               cpy_len * sizeof(IN_DT), 0);
      dt_offset = 0;
    } else if (sizeof(IN_DT) == sizeof(half)) {
      __memcpy(x1, input_score_ptr + input_offset + i * max_seg_iou_compute,
               cpy_len * sizeof(IN_DT), load_dir, cpy_len * sizeof(IN_DT),
               cpy_len * sizeof(IN_DT), 0);
      __bang_half2float((float *)score, (half *)x1, seg_len);
      dt_offset = max_seg_iou_compute;
    }
#if __BANG_ARCH__ >= 300
    __memcpy(inter_x1 + dt_offset,
             input_x1_ptr + input_offset + i * max_seg_iou_compute,
             cpy_len * sizeof(IN_DT), load_dir, max_seg_pad * sizeof(IN_DT),
             input_num_boxes * sizeof(IN_DT), 3);

    if (sizeof(IN_DT) == sizeof(half)) {
      __bang_half2float((float *)inter_x1,
                        (half *)inter_x1 + max_seg_iou_compute, seg_len);
      __bang_half2float((float *)inter_y1,
                        (half *)inter_y1 + max_seg_iou_compute, seg_len);
      __bang_half2float((float *)inter_x2,
                        (half *)inter_x2 + max_seg_iou_compute, seg_len);
      __bang_half2float((float *)inter_y2,
                        (half *)inter_y2 + max_seg_iou_compute, seg_len);
    }
    // box transfer
    __bang_minequal((float *)x1, (float *)inter_x1, (float *)inter_x2, seg_len);
    __bang_maxequal((float *)x2, (float *)inter_x1, (float *)inter_x2, seg_len);
    __bang_minequal((float *)y1, (float *)inter_y1, (float *)inter_y2, seg_len);
    __bang_maxequal((float *)y2, (float *)inter_y1, (float *)inter_y2, seg_len);
    // 1、 compute IOU
    // get the area_I
    __bang_maxeq_scalar((float *)inter_x1, (float *)x1, max_box_x1,
                        seg_len);  // inter_x1
    __bang_mineq_scalar((float *)inter_x2, (float *)x2, max_box_x2,
                        seg_len);  // inter_x2
    __bang_sub((float *)inter_x1, (float *)inter_x2, (float *)inter_x1,
               seg_len);
    if (algo == 1 && offset != 0.0) {
      __bang_add_scalar((float *)inter_x1, (float *)inter_x1, offset, seg_len);
    }
    computeReluN((float *)inter_x1, (float *)inter_x1, NULL,
                 seg_len);  // inter_w
    __bang_maxeq_scalar((float *)inter_y1, (float *)y1, float(max_box_y1),
                        seg_len);  // inter_y1
    __bang_mineq_scalar((float *)inter_y2, (float *)y2, float(max_box_y2),
                        seg_len);  // inter_y2
    __bang_sub((float *)inter_y1, (float *)inter_y2, (float *)inter_y1,
               seg_len);
    if (algo == 1 && offset != 0.0) {
      __bang_add_scalar((float *)inter_y1, (float *)inter_y1, offset, seg_len);
    }
    computeReluN((float *)inter_y1, (float *)inter_y1, NULL,
                 seg_len);  // inter_h
    __bang_mul((float *)inter_x1, (float *)inter_x1, (float *)inter_y1,
               seg_len);  // area_I
    // get the area of input_box: area = (x2 - x1) * (y2 - y1);
    if (algo == 1 && offset != 0.0) {
      __bang_fusion(FUSION_FSA, (float *)inter_y1, (float *)x2, (float *)x1,
                    offset, seg_len, seg_len);
      __bang_fusion(FUSION_FSA, (float *)inter_y2, (float *)y2, (float *)y1,
                    offset, seg_len, seg_len);
      __bang_mul((float *)inter_x2, (float *)inter_y1, (float *)inter_y2,
                 seg_len);  // area
    } else {
      __bang_sub((float *)inter_y1, (float *)x2, (float *)x1, seg_len);
      __bang_fusion(FUSION_FSM, (float *)inter_x2, (float *)y2, (float *)y1,
                    (float *)inter_y1, seg_len, seg_len);
    }
    // get the area_U: area + max_area - area_I
    __bang_fusion(FUSION_FAS, (float *)inter_x2, (float *)inter_x2, max_area,
                  (float *)inter_x1, seg_len, seg_len);
    // 2、 select the box
    // if IOU greater than thres, set the score to zero, abort it: area_U >
    // area_I * (1 / thresh)?
    if (thresh_iou > 0.0) {
      __bang_mul_scalar((float *)inter_x1, (float *)inter_x1, div_thresh_iou,
                        seg_len);
    } else {
      __bang_mul_scalar((float *)inter_x2, (float *)inter_x2, thresh_iou,
                        seg_len);
    }
    // process for nan
    __bang_lt((float *)inter_x1, (float *)inter_x2, (float *)inter_x1, seg_len);
    __bang_not((float *)inter_x1, (float *)inter_x1, seg_len);
    __bang_mul((float *)score, (float *)score, (float *)inter_x1, seg_len);
/******NMS COMPUTE END******/
#else
    __memcpy(x1 + dt_offset,
             input_x1_ptr + input_offset + i * max_seg_iou_compute,
             cpy_len * sizeof(IN_DT), load_dir, max_seg_pad * sizeof(IN_DT),
             input_num_boxes * sizeof(IN_DT), 3);
    if (sizeof(IN_DT) == sizeof(half)) {
      __bang_half2float((float *)x1, (half *)x1 + max_seg_iou_compute, seg_len);
      __bang_half2float((float *)y1, (half *)y1 + max_seg_iou_compute, seg_len);
      __bang_half2float((float *)x2, (half *)x2 + max_seg_iou_compute, seg_len);
      __bang_half2float((float *)y2, (half *)y2 + max_seg_iou_compute, seg_len);
    }
    // 1、 compute IOU
    // get the area_I
    __bang_write_value((float *)inter_y1, seg_len,
                       float(max_box[1]));  // max_x1
    __bang_maxequal((float *)inter_x1, (float *)x1, (float *)inter_y1,
                    seg_len);  // inter_x1
    __bang_write_value((float *)inter_y2, seg_len,
                       float(max_box[3]));  // max_x2
    __bang_minequal((float *)inter_x2, (float *)x2, (float *)inter_y2,
                    seg_len);  // inter_x2
    __bang_sub((float *)inter_x1, (float *)inter_x2, (float *)inter_x1,
               seg_len);
    if (algo == 1 && offset != 0.0) {
      __bang_add_scalar((float *)inter_x1, (float *)inter_x1, offset, seg_len);
    }
    computeReluN((float *)inter_x1, (float *)inter_x1, NULL,
                 seg_len);  // inter_w
    __bang_write_value((float *)inter_x2, seg_len,
                       float(max_box[2]));  // max_y1
    __bang_maxequal((float *)inter_y1, (float *)y1, (float *)inter_x2,
                    seg_len);  // inter_y1
    __bang_write_value((float *)inter_x2, seg_len,
                       float(max_box[4]));  // max_y2
    __bang_minequal((float *)inter_y2, (float *)y2, (float *)inter_x2,
                    seg_len);  // inter_y2
    __bang_sub((float *)inter_y1, (float *)inter_y2, (float *)inter_y1,
               seg_len);
    if (algo == 1 && offset != 0.0) {
      __bang_add_scalar((float *)inter_y1, (float *)inter_y1, offset, seg_len);
    }
    computeReluN((float *)inter_y1, (float *)inter_y1, NULL,
                 seg_len);  // inter_h
    __bang_mul((float *)inter_x1, (float *)inter_x1, (float *)inter_y1,
               seg_len);  // area_I
    // get the area of input_box: area = (x2 - x1) * (y2 - y1);
    __bang_sub((float *)inter_y1, (float *)x2, (float *)x1, seg_len);
    __bang_sub((float *)inter_y2, (float *)y2, (float *)y1, seg_len);
    if (algo == 1 && offset != 0.0) {
      __bang_add_scalar((float *)inter_y1, (float *)inter_y1, offset, seg_len);
      __bang_add_scalar((float *)inter_y2, (float *)inter_y2, offset, seg_len);
    }
    __bang_mul((float *)inter_x2, (float *)inter_y1, (float *)inter_y2,
               seg_len);  // area
    // get the area_U: area + max_area - area_I
    __bang_add_scalar((float *)inter_x2, (float *)inter_x2, float(max_area),
                      seg_len);
    __bang_sub((float *)inter_x2, (float *)inter_x2, (float *)inter_x1,
               seg_len);  // area_U
    // 2、 select the box
    // if IOU greater than thresh, set the score to zero, abort it: area_U >
    // area_I * (1 / thresh)?
    if (thresh_iou > 0.0) {
      __bang_mul_scalar((float *)inter_x1, (float *)inter_x1, div_thresh_iou,
                        seg_len);
    } else {
      __bang_mul_scalar((float *)inter_x2, (float *)inter_x2, thresh_iou,
                        seg_len);
    }
    __bang_ge((float *)inter_x1, (float *)inter_x2, (float *)inter_x1, seg_len);
    __bang_mul((float *)score, (float *)score, (float *)inter_x1, seg_len);
/******NMS COMPUTE END******/
#endif
    // update the score
    if (sizeof(IN_DT) == sizeof(half)) {
      convertFloat2half((half *)score, (float *)score, seg_len);
    }
    pvLock();
    __memcpy(input_score_ptr + input_offset + i * max_seg_iou_compute, score,
             cpy_len * sizeof(IN_DT), store_dir, cpy_len * sizeof(IN_DT),
             cpy_len * sizeof(IN_DT), 0);
    pvUnlock();
  }
}

#endif  // NMS_UTILS_HPP_
