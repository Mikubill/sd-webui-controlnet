// Copyright (c) OpenMMLab. All rights reserved
#include <torch/extension.h>

#include "pytorch_cpp_helper.hpp"

std::string get_compiler_version();
std::string get_compiling_cuda_version();

void assign_score_withk_forward(const Tensor &points, const Tensor &centers,
                                const Tensor &scores, const Tensor &knn_idx,
                                Tensor &output, int B, int N0, int N1, int M,
                                int K, int O, int aggregate);

void assign_score_withk_backward(const Tensor &grad_out, const Tensor &points,
                                 const Tensor &centers, const Tensor &scores,
                                 const Tensor &knn_idx, Tensor &grad_points,
                                 Tensor &grad_centers, Tensor &grad_scores,
                                 int B, int N0, int N1, int M, int K, int O,
                                 int aggregate);

void carafe_naive_forward(Tensor features, Tensor masks, Tensor output,
                          int kernel_size, int group_size, int scale_factor);

void carafe_naive_backward(Tensor top_grad, Tensor features, Tensor masks,
                           Tensor bottom_grad, Tensor mask_grad,
                           int kernel_size, int group_size, int scale_factor);

void carafe_forward(Tensor features, Tensor masks, Tensor rfeatures,
                    Tensor routput, Tensor rmasks, Tensor output,
                    int kernel_size, int group_size, int scale_factor);

void carafe_backward(Tensor top_grad, Tensor rfeatures, Tensor masks,
                     Tensor rtop_grad, Tensor rbottom_grad_hs,
                     Tensor rbottom_grad, Tensor rmask_grad, Tensor bottom_grad,
                     Tensor mask_grad, int kernel_size, int group_size,
                     int scale_factor);

void deform_conv_forward(Tensor input, Tensor weight, Tensor offset,
                         Tensor output, Tensor columns, Tensor ones, int kW,
                         int kH, int dW, int dH, int padW, int padH,
                         int dilationW, int dilationH, int group,
                         int deformable_group, int im2col_step);

void deform_conv_backward_input(Tensor input, Tensor offset, Tensor gradOutput,
                                Tensor gradInput, Tensor gradOffset,
                                Tensor weight, Tensor columns, int kW, int kH,
                                int dW, int dH, int padW, int padH,
                                int dilationW, int dilationH, int group,
                                int deformable_group, int im2col_step);

void deform_conv_backward_parameters(Tensor input, Tensor offset,
                                     Tensor gradOutput, Tensor gradWeight,
                                     Tensor columns, Tensor ones, int kW,
                                     int kH, int dW, int dH, int padW, int padH,
                                     int dilationW, int dilationH, int group,
                                     int deformable_group, float scale,
                                     int im2col_step);

void deform_roi_pool_forward(Tensor input, Tensor rois, Tensor offset,
                             Tensor output, int pooled_height, int pooled_width,
                             float spatial_scale, int sampling_ratio,
                             float gamma);

void deform_roi_pool_backward(Tensor grad_output, Tensor input, Tensor rois,
                              Tensor offset, Tensor grad_input,
                              Tensor grad_offset, int pooled_height,
                              int pooled_width, float spatial_scale,
                              int sampling_ratio, float gamma);

void group_points_forward(Tensor points_tensor, Tensor idx_tensor,
                          Tensor out_tensor, int b, int c, int n, int npoints,
                          int nsample);

void group_points_backward(Tensor grad_out_tensor, Tensor idx_tensor,
                           Tensor grad_points_tensor, int b, int c, int n,
                           int npoints, int nsample);

void stack_group_points_forward(Tensor features_tensor,
                                Tensor features_batch_cnt_tensor,
                                Tensor idx_tensor, Tensor idx_batch_cnt_tensor,
                                Tensor out_tensor, int b, int c, int m,
                                int nsample);

void stack_group_points_backward(Tensor grad_out_tensor, Tensor idx_tensor,
                                 Tensor idx_batch_cnt_tensor,
                                 Tensor features_batch_cnt_tensor,
                                 Tensor grad_features_tensor, int b, int c,
                                 int m, int n, int nsample);

void roipoint_pool3d_forward(Tensor xyz, Tensor boxes3d, Tensor pts_feature,
                             Tensor pooled_features, Tensor pooled_empty_flag);

void gather_points_forward(Tensor points_tensor, Tensor idx_tensor,
                           Tensor out_tensor, int b, int c, int n, int npoints);

void gather_points_backward(Tensor grad_out_tensor, Tensor idx_tensor,
                            Tensor grad_points_tensor, int b, int c, int n,
                            int npoints);

void sigmoid_focal_loss_forward(Tensor input, Tensor target, Tensor weight,
                                Tensor output, float gamma, float alpha);

void sigmoid_focal_loss_backward(Tensor input, Tensor target, Tensor weight,
                                 Tensor grad_input, float gamma, float alpha);

void softmax_focal_loss_forward(Tensor input, Tensor target, Tensor weight,
                                Tensor output, float gamma, float alpha);

void softmax_focal_loss_backward(Tensor input, Tensor target, Tensor weight,
                                 Tensor buff, Tensor grad_input, float gamma,
                                 float alpha);

void three_interpolate_forward(Tensor points_tensor, Tensor idx_tensor,
                               Tensor weight_tensor, Tensor out_tensor, int b,
                               int c, int m, int n);

void three_interpolate_backward(Tensor grad_out_tensor, Tensor idx_tensor,
                                Tensor weight_tensor, Tensor grad_points_tensor,
                                int b, int c, int n, int m);

void three_nn_forward(Tensor unknown_tensor, Tensor known_tensor,
                      Tensor dist2_tensor, Tensor idx_tensor, int b, int n,
                      int m);

void bbox_overlaps(const Tensor bboxes1, const Tensor bboxes2, Tensor ious,
                   const int mode, const bool aligned, const int offset);

void knn_forward(Tensor xyz_tensor, Tensor new_xyz_tensor, Tensor idx_tensor,
                 Tensor dist2_tensor, int b, int n, int m, int nsample);

void iou3d_boxes_overlap_bev_forward(Tensor boxes_a, Tensor boxes_b,
                                     Tensor ans_overlap);

void iou3d_nms3d_forward(Tensor boxes, Tensor keep, Tensor keep_num,
                         float nms_overlap_thresh);

void iou3d_nms3d_normal_forward(Tensor boxes, Tensor keep, Tensor keep_num,
                                float nms_overlap_thresh);

void furthest_point_sampling_forward(Tensor points_tensor, Tensor temp_tensor,
                                     Tensor idx_tensor, int b, int n, int m);

void furthest_point_sampling_with_dist_forward(Tensor points_tensor,
                                               Tensor temp_tensor,
                                               Tensor idx_tensor, int b, int n,
                                               int m);

void masked_im2col_forward(const Tensor im, const Tensor mask_h_idx,
                           const Tensor mask_w_idx, Tensor col,
                           const int kernel_h, const int kernel_w,
                           const int pad_h, const int pad_w);

void masked_col2im_forward(const Tensor col, const Tensor mask_h_idx,
                           const Tensor mask_w_idx, Tensor im, int height,
                           int width, int channels);

void modulated_deform_conv_forward(
    Tensor input, Tensor weight, Tensor bias, Tensor ones, Tensor offset,
    Tensor mask, Tensor output, Tensor columns, int kernel_h, int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, const int group,
    const int deformable_group, const bool with_bias);

void modulated_deform_conv_backward(
    Tensor input, Tensor weight, Tensor bias, Tensor ones, Tensor offset,
    Tensor mask, Tensor columns, Tensor grad_input, Tensor grad_weight,
    Tensor grad_bias, Tensor grad_offset, Tensor grad_mask, Tensor grad_output,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
    int pad_w, int dilation_h, int dilation_w, int group, int deformable_group,
    const bool with_bias);

Tensor ms_deform_attn_forward(const Tensor &value, const Tensor &spatial_shapes,
                              const Tensor &level_start_index,
                              const Tensor &sampling_loc,
                              const Tensor &attn_weight, const int im2col_step);

void ms_deform_attn_backward(const Tensor &value, const Tensor &spatial_shapes,
                             const Tensor &level_start_index,
                             const Tensor &sampling_loc,
                             const Tensor &attn_weight,
                             const Tensor &grad_output, Tensor &grad_value,
                             Tensor &grad_sampling_loc,
                             Tensor &grad_attn_weight, const int im2col_step);

Tensor nms(Tensor boxes, Tensor scores, float iou_threshold, int offset);

Tensor softnms(Tensor boxes, Tensor scores, Tensor dets, float iou_threshold,
               float sigma, float min_score, int method, int offset);

std::vector<std::vector<int>> nms_match(Tensor dets, float iou_threshold);

std::vector<std::vector<float>> pixel_group(
    Tensor score, Tensor mask, Tensor embedding, Tensor kernel_label,
    Tensor kernel_contour, int kernel_region_num, float distance_threshold);

std::vector<std::vector<int>> contour_expand(Tensor kernel_mask,
                                             Tensor internal_kernel_label,
                                             int min_kernel_area,
                                             int kernel_num);

void roi_align_forward(Tensor input, Tensor rois, Tensor output,
                       Tensor argmax_y, Tensor argmax_x, int aligned_height,
                       int aligned_width, float spatial_scale,
                       int sampling_ratio, int pool_mode, bool aligned);

void roi_align_backward(Tensor grad_output, Tensor rois, Tensor argmax_y,
                        Tensor argmax_x, Tensor grad_input, int aligned_height,
                        int aligned_width, float spatial_scale,
                        int sampling_ratio, int pool_mode, bool aligned);

void roi_pool_forward(Tensor input, Tensor rois, Tensor output, Tensor argmax,
                      int pooled_height, int pooled_width, float spatial_scale);

void roi_pool_backward(Tensor grad_output, Tensor rois, Tensor argmax,
                       Tensor grad_input, int pooled_height, int pooled_width,
                       float spatial_scale);

void sync_bn_forward_mean(const Tensor input, Tensor mean);

void sync_bn_forward_var(const Tensor input, const Tensor mean, Tensor var);

void sync_bn_forward_output(const Tensor input, const Tensor mean,
                            const Tensor var, const Tensor weight,
                            const Tensor bias, Tensor running_mean,
                            Tensor running_var, Tensor norm, Tensor std,
                            Tensor output, float eps, float momentum,
                            int group_size);

void sync_bn_backward_param(const Tensor grad_output, const Tensor norm,
                            Tensor grad_weight, Tensor grad_bias);

void sync_bn_backward_data(const Tensor grad_output, const Tensor weight,
                           const Tensor grad_weight, const Tensor grad_bias,
                           const Tensor norm, const Tensor std,
                           Tensor grad_input);

void psamask_forward(const Tensor input, Tensor output, const int psa_type,
                     const int num_, const int h_feature, const int w_feature,
                     const int h_mask, const int w_mask, const int half_h_mask,
                     const int half_w_mask);

void psamask_backward(Tensor grad_output, const Tensor grad_input,
                      const int psa_type, const int num_, const int h_feature,
                      const int w_feature, const int h_mask, const int w_mask,
                      const int half_h_mask, const int half_w_mask);

void tin_shift_forward(Tensor input, Tensor shift, Tensor output);

void tin_shift_backward(Tensor grad_output, Tensor shift, Tensor grad_input);

void ball_query_forward(Tensor new_xyz_tensor, Tensor xyz_tensor,
                        Tensor idx_tensor, int b, int n, int m,
                        float min_radius, float max_radius, int nsample);

void stack_ball_query_forward(Tensor new_xyz_tensor, Tensor new_xyz_batch_cnt,
                              Tensor xyz_tensor, Tensor xyz_batch_cnt,
                              Tensor idx_tensor, float max_radius, int nsample);

void prroi_pool_forward(Tensor input, Tensor rois, Tensor output,
                        int pooled_height, int pooled_width,
                        float spatial_scale);

void prroi_pool_backward(Tensor grad_output, Tensor rois, Tensor grad_input,
                         int pooled_height, int pooled_width,
                         float spatial_scale);

void prroi_pool_coor_backward(Tensor output, Tensor grad_output, Tensor input,
                              Tensor rois, Tensor grad_rois, int pooled_height,
                              int pooled_width, float spatial_scale);

template <unsigned NDim>
std::vector<torch::Tensor> get_indice_pairs_forward(
    torch::Tensor indices, int64_t batchSize,
    std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
    std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose);

template <unsigned NDim>
std::vector<Tensor> get_indice_pairs_backward(
    Tensor indices, Tensor gridOut, int64_t batchSize,
    std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
    std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose);

Tensor indice_conv_forward(Tensor features, Tensor filters, Tensor indicePairs,
                           Tensor indiceNum, int64_t numActOut,
                           int64_t _inverse, int64_t _subM);

std::vector<Tensor> indice_conv_backward(Tensor features, Tensor filters,
                                         Tensor outGrad, Tensor indicePairs,
                                         Tensor indiceNum, int64_t _inverse,
                                         int64_t _subM);

Tensor fused_indice_conv_batchnorm_forward(Tensor features, Tensor filters,
                                           Tensor bias, Tensor indicePairs,
                                           Tensor indiceNum, int64_t numActOut,
                                           int64_t _inverse, int64_t _subM);

Tensor indice_maxpool_forward(Tensor features, Tensor indicePairs,
                              Tensor indiceNum, int64_t numAct);

Tensor indice_maxpool_backward(Tensor features, Tensor outFeatures,
                               Tensor outGrad, Tensor indicePairs,
                               Tensor indiceNum);

void box_iou_rotated(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                     const int mode_flag, const bool aligned);

Tensor nms_rotated(const Tensor dets, const Tensor scores, const Tensor order,
                   const Tensor dets_sorted, const Tensor labels,
                   const float iou_threshold, const int multi_label);

Tensor upfirdn2d(const Tensor &input, const Tensor &kernel, int up_x, int up_y,
                 int down_x, int down_y, int pad_x0, int pad_x1, int pad_y0,
                 int pad_y1);

Tensor fused_bias_leakyrelu(const Tensor &input, const Tensor &bias,
                            const Tensor &refer, int act, int grad, float alpha,
                            float scale);

void roi_align_rotated_forward(Tensor input, Tensor rois, Tensor output,
                               int pooled_height, int pooled_width,
                               float spatial_scale, int sampling_ratio,
                               bool aligned, bool clockwise);

void roi_align_rotated_backward(Tensor grad_output, Tensor rois,
                                Tensor grad_input, int pooled_height,
                                int pooled_width, float spatial_scale,
                                int sampling_ratio, bool aligned,
                                bool clockwise);

std::vector<torch::Tensor> dynamic_point_to_voxel_forward(
    const torch::Tensor &feats, const torch::Tensor &coors,
    const std::string &reduce_type);

void dynamic_point_to_voxel_backward(torch::Tensor &grad_feats,
                                     const torch::Tensor &grad_reduced_feats,
                                     const torch::Tensor &feats,
                                     const torch::Tensor &reduced_feats,
                                     const torch::Tensor &coors_idx,
                                     const torch::Tensor &reduce_count,
                                     const std::string &reduce_type);

void hard_voxelize_forward(const at::Tensor &points,
                           const at::Tensor &voxel_size,
                           const at::Tensor &coors_range, at::Tensor &voxels,
                           at::Tensor &coors, at::Tensor &num_points_per_voxel,
                           at::Tensor &voxel_num, const int max_points,
                           const int max_voxels, const int NDim,
                           const bool deterministic);

void dynamic_voxelize_forward(const at::Tensor &points,
                              const at::Tensor &voxel_size,
                              const at::Tensor &coors_range, at::Tensor &coors,
                              const int NDim);

void border_align_forward(const Tensor &input, const Tensor &boxes,
                          Tensor output, Tensor argmax_idx,
                          const int pool_size);

void border_align_backward(const Tensor &grad_output, const Tensor &boxes,
                           const Tensor &argmax_idx, Tensor grad_input,
                           const int pool_size);

void points_in_boxes_cpu_forward(Tensor boxes_tensor, Tensor pts_tensor,
                                 Tensor pts_indices_tensor);

void points_in_boxes_part_forward(Tensor boxes_tensor, Tensor pts_tensor,
                                  Tensor box_idx_of_points_tensor);

void points_in_boxes_all_forward(Tensor boxes_tensor, Tensor pts_tensor,
                                 Tensor box_idx_of_points_tensor);

void roiaware_pool3d_forward(Tensor rois, Tensor pts, Tensor pts_feature,
                             Tensor argmax, Tensor pts_idx_of_voxels,
                             Tensor pooled_features, int pool_method);

void roiaware_pool3d_backward(Tensor pts_idx_of_voxels, Tensor argmax,
                              Tensor grad_out, Tensor grad_in, int pool_method);

void correlation_forward(Tensor input1, Tensor input2, Tensor output, int kH,
                         int kW, int patchH, int patchW, int padH, int padW,
                         int dilationH, int dilationW, int dilation_patchH,
                         int dilation_patchW, int dH, int dW);

void correlation_backward(Tensor grad_output, Tensor input1, Tensor input2,
                          Tensor grad_input1, Tensor grad_input2, int kH,
                          int kW, int patchH, int patchW, int padH, int padW,
                          int dilationH, int dilationW, int dilation_patchH,
                          int dilation_patchW, int dH, int dW);

void rotated_feature_align_forward(const Tensor features,
                                   const Tensor best_bboxes, Tensor output,
                                   const float spatial_scale, const int points);

void rotated_feature_align_backward(const Tensor top_grad,
                                    const Tensor best_bboxes,
                                    Tensor bottom_grad,
                                    const float spatial_scale,
                                    const int points);

void riroi_align_rotated_forward(Tensor features, Tensor rois, Tensor output,
                                 int pooled_height, int pooled_width,
                                 float spatial_scale, int num_samples,
                                 int num_orientations, bool clockwise);

void riroi_align_rotated_backward(Tensor top_grad, Tensor rois,
                                  Tensor bottom_grad, int pooled_height,
                                  int pooled_width, float spatial_scale,
                                  int num_samples, int num_orientations,
                                  bool clockwise);

void points_in_polygons_forward(Tensor points, Tensor polygons, Tensor output);

void min_area_polygons(const Tensor pointsets, Tensor polygons);

void active_rotated_filter_forward(const Tensor input, const Tensor indices,
                                   Tensor output);

void active_rotated_filter_backward(const Tensor grad_out, const Tensor indices,
                                    Tensor grad_in);

void convex_iou(const Tensor pointsets, const Tensor polygons, Tensor ious);

void convex_giou(const Tensor pointsets, const Tensor polygons, Tensor output);

at::Tensor diff_iou_rotated_sort_vertices_forward(at::Tensor vertices,
                                                  at::Tensor mask,
                                                  at::Tensor num_valid);

void chamfer_distance_forward(const Tensor xyz1, const Tensor xyz2,
                              const Tensor dist1, const Tensor dist2,
                              const Tensor idx1, const Tensor idx);

void chamfer_distance_backward(const Tensor xyz1, const Tensor xyz2,
                               Tensor idx1, Tensor idx2, Tensor graddist1,
                               Tensor graddist2, Tensor gradxyz1,
                               Tensor gradxyz2);

void box_iou_quadri(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                    const int mode_flag, const bool aligned);

Tensor nms_quadri(const Tensor dets, const Tensor scores, const Tensor order,
                  const Tensor dets_sorted, const float iou_threshold,
                  const int multi_label);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("upfirdn2d", &upfirdn2d, "upfirdn2d (CUDA)", py::arg("input"),
        py::arg("kernel"), py::arg("up_x"), py::arg("up_y"), py::arg("down_x"),
        py::arg("down_y"), py::arg("pad_x0"), py::arg("pad_x1"),
        py::arg("pad_y0"), py::arg("pad_y1"));
  m.def("fused_bias_leakyrelu", &fused_bias_leakyrelu,
        "fused_bias_leakyrelu (CUDA)", py::arg("input"), py::arg("bias"),
        py::arg("empty"), py::arg("act"), py::arg("grad"), py::arg("alpha"),
        py::arg("scale"));
  m.def("gather_points_forward", &gather_points_forward,
        "gather_points_forward", py::arg("points_tensor"),
        py::arg("idx_tensor"), py::arg("out_tensor"), py::arg("b"),
        py::arg("c"), py::arg("n"), py::arg("npoints"));
  m.def("gather_points_backward", &gather_points_backward,
        "gather_points_backward", py::arg("grad_out_tensor"),
        py::arg("idx_tensor"), py::arg("grad_points_tensor"), py::arg("b"),
        py::arg("c"), py::arg("n"), py::arg("npoints"));
  m.def("get_compiler_version", &get_compiler_version, "get_compiler_version");
  m.def("get_compiling_cuda_version", &get_compiling_cuda_version,
        "get_compiling_cuda_version");
  m.def("assign_score_withk_forward", &assign_score_withk_forward,
        "assign_score_withk_forward", py::arg("points"), py::arg("centers"),
        py::arg("scores"), py::arg("knn_idx"), py::arg("output"), py::arg("B"),
        py::arg("N0"), py::arg("N1"), py::arg("M"), py::arg("K"), py::arg("O"),
        py::arg("aggregate"));
  m.def("assign_score_withk_backward", &assign_score_withk_backward,
        "assign_score_withk_backward", py::arg("grad_out"), py::arg("points"),
        py::arg("centers"), py::arg("scores"), py::arg("knn_idx"),
        py::arg("grad_points"), py::arg("grad_centers"), py::arg("grad_scores"),
        py::arg("B"), py::arg("N0"), py::arg("N1"), py::arg("M"), py::arg("K"),
        py::arg("O"), py::arg("aggregate"));
  m.def("knn_forward", &knn_forward, "knn_forward", py::arg("xyz_tensor"),
        py::arg("new_xyz_tensor"), py::arg("idx_tensor"),
        py::arg("dist2_tensor"), py::arg("b"), py::arg("n"), py::arg("m"),
        py::arg("nsample"));
  m.def("carafe_naive_forward", &carafe_naive_forward, "carafe_naive_forward",
        py::arg("features"), py::arg("masks"), py::arg("output"),
        py::arg("kernel_size"), py::arg("group_size"), py::arg("scale_factor"));
  m.def("carafe_naive_backward", &carafe_naive_backward,
        "carafe_naive_backward", py::arg("top_grad"), py::arg("features"),
        py::arg("masks"), py::arg("bottom_grad"), py::arg("mask_grad"),
        py::arg("kernel_size"), py::arg("group_size"), py::arg("scale_factor"));
  m.def("carafe_forward", &carafe_forward, "carafe_forward",
        py::arg("features"), py::arg("masks"), py::arg("rfeatures"),
        py::arg("routput"), py::arg("rmasks"), py::arg("output"),
        py::arg("kernel_size"), py::arg("group_size"), py::arg("scale_factor"));
  m.def("carafe_backward", &carafe_backward, "carafe_backward",
        py::arg("top_grad"), py::arg("rfeatures"), py::arg("masks"),
        py::arg("rtop_grad"), py::arg("rbottom_grad_hs"),
        py::arg("rbottom_grad"), py::arg("rmask_grad"), py::arg("bottom_grad"),
        py::arg("mask_grad"), py::arg("kernel_size"), py::arg("group_size"),
        py::arg("scale_factor"));
  m.def("deform_conv_forward", &deform_conv_forward, "deform_conv_forward",
        py::arg("input"), py::arg("weight"), py::arg("offset"),
        py::arg("output"), py::arg("columns"), py::arg("ones"), py::arg("kW"),
        py::arg("kH"), py::arg("dW"), py::arg("dH"), py::arg("padW"),
        py::arg("padH"), py::arg("dilationW"), py::arg("dilationH"),
        py::arg("group"), py::arg("deformable_group"), py::arg("im2col_step"));
  m.def("deform_conv_backward_input", &deform_conv_backward_input,
        "deform_conv_backward_input", py::arg("input"), py::arg("offset"),
        py::arg("gradOutput"), py::arg("gradInput"), py::arg("gradOffset"),
        py::arg("weight"), py::arg("columns"), py::arg("kW"), py::arg("kH"),
        py::arg("dW"), py::arg("dH"), py::arg("padW"), py::arg("padH"),
        py::arg("dilationW"), py::arg("dilationH"), py::arg("group"),
        py::arg("deformable_group"), py::arg("im2col_step"));
  m.def("deform_conv_backward_parameters", &deform_conv_backward_parameters,
        "deform_conv_backward_parameters", py::arg("input"), py::arg("offset"),
        py::arg("gradOutput"), py::arg("gradWeight"), py::arg("columns"),
        py::arg("ones"), py::arg("kW"), py::arg("kH"), py::arg("dW"),
        py::arg("dH"), py::arg("padW"), py::arg("padH"), py::arg("dilationW"),
        py::arg("dilationH"), py::arg("group"), py::arg("deformable_group"),
        py::arg("scale"), py::arg("im2col_step"));
  m.def("deform_roi_pool_forward", &deform_roi_pool_forward,
        "deform roi pool forward", py::arg("input"), py::arg("rois"),
        py::arg("offset"), py::arg("output"), py::arg("pooled_height"),
        py::arg("pooled_width"), py::arg("spatial_scale"),
        py::arg("sampling_ratio"), py::arg("gamma"));
  m.def("deform_roi_pool_backward", &deform_roi_pool_backward,
        "deform roi pool backward", py::arg("grad_output"), py::arg("input"),
        py::arg("rois"), py::arg("offset"), py::arg("grad_input"),
        py::arg("grad_offset"), py::arg("pooled_height"),
        py::arg("pooled_width"), py::arg("spatial_scale"),
        py::arg("sampling_ratio"), py::arg("gamma"));
  m.def("roipoint_pool3d_forward", &roipoint_pool3d_forward,
        "roipoint_pool3d_forward", py::arg("xyz"), py::arg("boxes3d"),
        py::arg("pts_feature"), py::arg("pooled_features"),
        py::arg("pooled_empty_flag"));
  m.def("sigmoid_focal_loss_forward", &sigmoid_focal_loss_forward,
        "sigmoid_focal_loss_forward ", py::arg("input"), py::arg("target"),
        py::arg("weight"), py::arg("output"), py::arg("gamma"),
        py::arg("alpha"));
  m.def("sigmoid_focal_loss_backward", &sigmoid_focal_loss_backward,
        "sigmoid_focal_loss_backward", py::arg("input"), py::arg("target"),
        py::arg("weight"), py::arg("grad_input"), py::arg("gamma"),
        py::arg("alpha"));
  m.def("softmax_focal_loss_forward", &softmax_focal_loss_forward,
        "softmax_focal_loss_forward", py::arg("input"), py::arg("target"),
        py::arg("weight"), py::arg("output"), py::arg("gamma"),
        py::arg("alpha"));
  m.def("softmax_focal_loss_backward", &softmax_focal_loss_backward,
        "softmax_focal_loss_backward", py::arg("input"), py::arg("target"),
        py::arg("weight"), py::arg("buff"), py::arg("grad_input"),
        py::arg("gamma"), py::arg("alpha"));
  m.def("three_interpolate_forward", &three_interpolate_forward,
        "three_interpolate_forward", py::arg("points_tensor"),
        py::arg("idx_tensor"), py::arg("weight_tensor"), py::arg("out_tensor"),
        py::arg("b"), py::arg("c"), py::arg("m"), py::arg("n"));
  m.def("three_interpolate_backward", &three_interpolate_backward,
        "three_interpolate_backward", py::arg("grad_out_tensor"),
        py::arg("idx_tensor"), py::arg("weight_tensor"),
        py::arg("grad_points_tensor"), py::arg("b"), py::arg("c"), py::arg("n"),
        py::arg("m"));
  m.def("three_nn_forward", &three_nn_forward, "three_nn_forward",
        py::arg("unknown_tensor"), py::arg("known_tensor"),
        py::arg("dist2_tensor"), py::arg("idx_tensor"), py::arg("b"),
        py::arg("n"), py::arg("m"));
  m.def("bbox_overlaps", &bbox_overlaps, "bbox_overlaps", py::arg("bboxes1"),
        py::arg("bboxes2"), py::arg("ious"), py::arg("mode"),
        py::arg("aligned"), py::arg("offset"));
  m.def("group_points_forward", &group_points_forward, "group_points_forward",
        py::arg("points_tensor"), py::arg("idx_tensor"), py::arg("out_tensor"),
        py::arg("b"), py::arg("c"), py::arg("n"), py::arg("npoints"),
        py::arg("nsample"));
  m.def("group_points_backward", &group_points_backward,
        "group_points_backward", py::arg("grad_out_tensor"),
        py::arg("idx_tensor"), py::arg("grad_points_tensor"), py::arg("b"),
        py::arg("c"), py::arg("n"), py::arg("npoints"), py::arg("nsample"));
  m.def("stack_group_points_forward", &stack_group_points_forward,
        "stack_group_points_forward", py::arg("features_tensor"),
        py::arg("features_batch_cnt_tensor"), py::arg("idx_tensor"),
        py::arg("idx_batch_cnt_tensor"), py::arg("out_tensor"), py::arg("b"),
        py::arg("c"), py::arg("m"), py::arg("nsample"));
  m.def("stack_group_points_backward", &stack_group_points_backward,
        "stack_group_points_backward", py::arg("grad_out_tensor"),
        py::arg("idx_tensor"), py::arg("idx_batch_cnt_tensor"),
        py::arg("features_batch_cnt_tensor"), py::arg("grad_features_tensor"),
        py::arg("b"), py::arg("c"), py::arg("m"), py::arg("n"),
        py::arg("nsample"));
  m.def("knn_forward", &knn_forward, "knn_forward", py::arg("b"), py::arg("n"),
        py::arg("m"), py::arg("nsample"), py::arg("xyz_tensor"),
        py::arg("new_xyz_tensor"), py::arg("idx_tensor"),
        py::arg("dist2_tensor"));
  m.def("iou3d_boxes_overlap_bev_forward", &iou3d_boxes_overlap_bev_forward,
        "iou3d_boxes_overlap_bev_forward", py::arg("boxes_a"),
        py::arg("boxes_b"), py::arg("ans_iou"));
  m.def("iou3d_nms3d_forward", &iou3d_nms3d_forward, "iou3d_nms3d_forward",
        py::arg("boxes"), py::arg("keep"), py::arg("num_out"),
        py::arg("nms_overlap_thresh"));
  m.def("iou3d_nms3d_normal_forward", &iou3d_nms3d_normal_forward,
        "iou3d_nms3d_normal_forward", py::arg("boxes"), py::arg("keep"),
        py::arg("num_out"), py::arg("nms_overlap_thresh"));
  m.def("furthest_point_sampling_forward", &furthest_point_sampling_forward,
        "furthest_point_sampling_forward", py::arg("points_tensor"),
        py::arg("temp_tensor"), py::arg("idx_tensor"), py::arg("b"),
        py::arg("n"), py::arg("m"));
  m.def("furthest_point_sampling_with_dist_forward",
        &furthest_point_sampling_with_dist_forward,
        "furthest_point_sampling_with_dist_forward", py::arg("points_tensor"),
        py::arg("temp_tensor"), py::arg("idx_tensor"), py::arg("b"),
        py::arg("n"), py::arg("m"));
  m.def("masked_im2col_forward", &masked_im2col_forward,
        "masked_im2col_forward", py::arg("im"), py::arg("mask_h_idx"),
        py::arg("mask_w_idx"), py::arg("col"), py::arg("kernel_h"),
        py::arg("kernel_w"), py::arg("pad_h"), py::arg("pad_w"));
  m.def("masked_col2im_forward", &masked_col2im_forward,
        "masked_col2im_forward", py::arg("col"), py::arg("mask_h_idx"),
        py::arg("mask_w_idx"), py::arg("im"), py::arg("height"),
        py::arg("width"), py::arg("channels"));
  m.def("modulated_deform_conv_forward", &modulated_deform_conv_forward,
        "modulated deform conv forward", py::arg("input"), py::arg("weight"),
        py::arg("bias"), py::arg("ones"), py::arg("offset"), py::arg("mask"),
        py::arg("output"), py::arg("columns"), py::arg("kernel_h"),
        py::arg("kernel_w"), py::arg("stride_h"), py::arg("stride_w"),
        py::arg("pad_h"), py::arg("pad_w"), py::arg("dilation_h"),
        py::arg("dilation_w"), py::arg("group"), py::arg("deformable_group"),
        py::arg("with_bias"));
  m.def("modulated_deform_conv_backward", &modulated_deform_conv_backward,
        "modulated deform conv backward", py::arg("input"), py::arg("weight"),
        py::arg("bias"), py::arg("ones"), py::arg("offset"), py::arg("mask"),
        py::arg("columns"), py::arg("grad_input"), py::arg("grad_weight"),
        py::arg("grad_bias"), py::arg("grad_offset"), py::arg("grad_mask"),
        py::arg("grad_output"), py::arg("kernel_h"), py::arg("kernel_w"),
        py::arg("stride_h"), py::arg("stride_w"), py::arg("pad_h"),
        py::arg("pad_w"), py::arg("dilation_h"), py::arg("dilation_w"),
        py::arg("group"), py::arg("deformable_group"), py::arg("with_bias"));
  m.def("nms", &nms, "nms (CPU/CUDA) ", py::arg("boxes"), py::arg("scores"),
        py::arg("iou_threshold"), py::arg("offset"));
  m.def("softnms", &softnms, "softnms (CPU) ", py::arg("boxes"),
        py::arg("scores"), py::arg("dets"), py::arg("iou_threshold"),
        py::arg("sigma"), py::arg("min_score"), py::arg("method"),
        py::arg("offset"));
  m.def("nms_match", &nms_match, "nms_match (CPU) ", py::arg("dets"),
        py::arg("iou_threshold"));
  m.def("pixel_group", &pixel_group, "pixel group (CPU) ", py::arg("score"),
        py::arg("mask"), py::arg("embedding"), py::arg("kernel_label"),
        py::arg("kernel_contour"), py::arg("kernel_region_label"),
        py::arg("distance_threshold"));
  m.def("contour_expand", &contour_expand, "contour exapnd (CPU) ",
        py::arg("kernel_mask"), py::arg("internal_kernel_label"),
        py::arg("min_kernel_area"), py::arg("kernel_num"));
  m.def("roi_align_forward", &roi_align_forward, "roi_align forward",
        py::arg("input"), py::arg("rois"), py::arg("output"),
        py::arg("argmax_y"), py::arg("argmax_x"), py::arg("aligned_height"),
        py::arg("aligned_width"), py::arg("spatial_scale"),
        py::arg("sampling_ratio"), py::arg("pool_mode"), py::arg("aligned"));
  m.def("roi_align_backward", &roi_align_backward, "roi_align backward",
        py::arg("grad_output"), py::arg("rois"), py::arg("argmax_y"),
        py::arg("argmax_x"), py::arg("grad_input"), py::arg("aligned_height"),
        py::arg("aligned_width"), py::arg("spatial_scale"),
        py::arg("sampling_ratio"), py::arg("pool_mode"), py::arg("aligned"));
  m.def("roi_pool_forward", &roi_pool_forward, "roi_pool forward",
        py::arg("input"), py::arg("rois"), py::arg("output"), py::arg("argmax"),
        py::arg("pooled_height"), py::arg("pooled_width"),
        py::arg("spatial_scale"));
  m.def("roi_pool_backward", &roi_pool_backward, "roi_pool backward",
        py::arg("grad_output"), py::arg("rois"), py::arg("argmax"),
        py::arg("grad_input"), py::arg("pooled_height"),
        py::arg("pooled_width"), py::arg("spatial_scale"));
  m.def("sync_bn_forward_mean", &sync_bn_forward_mean, "sync_bn forward_mean",
        py::arg("input"), py::arg("mean"));
  m.def("sync_bn_forward_var", &sync_bn_forward_var, "sync_bn forward_var",
        py::arg("input"), py::arg("mean"), py::arg("var"));
  m.def("sync_bn_forward_output", &sync_bn_forward_output,
        "sync_bn forward_output", py::arg("input"), py::arg("mean"),
        py::arg("var"), py::arg("weight"), py::arg("bias"),
        py::arg("running_mean"), py::arg("running_var"), py::arg("norm"),
        py::arg("std"), py::arg("output"), py::arg("eps"), py::arg("momentum"),
        py::arg("group_size"));
  m.def("sync_bn_backward_param", &sync_bn_backward_param,
        "sync_bn backward_param", py::arg("grad_output"), py::arg("norm"),
        py::arg("grad_weight"), py::arg("grad_bias"));
  m.def("sync_bn_backward_data", &sync_bn_backward_data,
        "sync_bn backward_data", py::arg("grad_output"), py::arg("weight"),
        py::arg("grad_weight"), py::arg("grad_bias"), py::arg("norm"),
        py::arg("std"), py::arg("grad_input"));
  m.def("get_indice_pairs_2d_forward", &get_indice_pairs_forward<2>,
        "get_indice_pairs_2d_forward", py::arg("indices"), py::arg("batchSize"),
        py::arg("outSpatialShape"), py::arg("spatialShape"),
        py::arg("kernelSize"), py::arg("stride"), py::arg("padding"),
        py::arg("dilation"), py::arg("outPadding"), py::arg("_subM"),
        py::arg("_transpose"));
  m.def("get_indice_pairs_3d_forward", &get_indice_pairs_forward<3>,
        "get_indice_pairs_3d_forward", py::arg("indices"), py::arg("batchSize"),
        py::arg("outSpatialShape"), py::arg("spatialShape"),
        py::arg("kernelSize"), py::arg("stride"), py::arg("padding"),
        py::arg("dilation"), py::arg("outPadding"), py::arg("_subM"),
        py::arg("_transpose"));
  m.def("get_indice_pairs_4d_forward", &get_indice_pairs_forward<4>,
        "get_indice_pairs_4d_forward", py::arg("indices"), py::arg("batchSize"),
        py::arg("outSpatialShape"), py::arg("spatialShape"),
        py::arg("kernelSize"), py::arg("stride"), py::arg("padding"),
        py::arg("dilation"), py::arg("outPadding"), py::arg("_subM"),
        py::arg("_transpose"));
  m.def("get_indice_pairs_2d_backward", &get_indice_pairs_backward<2>,
        "get_indice_pairs_2d_backward", py::arg("indices"), py::arg("gridOut"),
        py::arg("batchSize"), py::arg("outSpatialShape"),
        py::arg("spatialShape"), py::arg("kernelSize"), py::arg("stride"),
        py::arg("padding"), py::arg("dilation"), py::arg("outPadding"),
        py::arg("_subM"), py::arg("_transpose"));
  m.def("get_indice_pairs_3d_backward", &get_indice_pairs_backward<3>,
        "get_indice_pairs_3d_backward", py::arg("indices"), py::arg("gridOut"),
        py::arg("batchSize"), py::arg("outSpatialShape"),
        py::arg("spatialShape"), py::arg("kernelSize"), py::arg("stride"),
        py::arg("padding"), py::arg("dilation"), py::arg("outPadding"),
        py::arg("_subM"), py::arg("_transpose"));
  m.def("indice_conv_forward", &indice_conv_forward, "indice_conv_forward",
        py::arg("features"), py::arg("filters"), py::arg("indicePairs"),
        py::arg("indiceNum"), py::arg("numActOut"), py::arg("_inverse"),
        py::arg("_subM"));
  m.def("indice_conv_backward", &indice_conv_backward, "indice_conv_backward",
        py::arg("features"), py::arg("filters"), py::arg("outGrad"),
        py::arg("indicePairs"), py::arg("indiceNum"), py::arg("_inverse"),
        py::arg("_subM"));
  m.def("fused_indice_conv_forward", &fused_indice_conv_batchnorm_forward,
        "fused_indice_conv_forward", py::arg("features"), py::arg("filters"),
        py::arg("bias"), py::arg("indicePairs"), py::arg("indiceNum"),
        py::arg("numActOut"), py::arg("_inverse"), py::arg("_subM"));
  m.def("indice_maxpool_forward", &indice_maxpool_forward,
        "indice_maxpool_forward", py::arg("features"), py::arg("indicePairs"),
        py::arg("indiceNum"), py::arg("numAct"));
  m.def("indice_maxpool_backward", &indice_maxpool_backward,
        "indice_maxpool_backward", py::arg("features"), py::arg("outFeatures"),
        py::arg("outGrad"), py::arg("indicePairs"), py::arg("indiceNum"));
  m.def("psamask_forward", &psamask_forward, "PSAMASK forward (CPU/CUDA)",
        py::arg("input"), py::arg("output"), py::arg("psa_type"),
        py::arg("num_"), py::arg("h_feature"), py::arg("w_feature"),
        py::arg("h_mask"), py::arg("w_mask"), py::arg("half_h_mask"),
        py::arg("half_w_mask"));
  m.def("psamask_backward", &psamask_backward, "PSAMASK backward (CPU/CUDA)",
        py::arg("grad_output"), py::arg("grad_input"), py::arg("psa_type"),
        py::arg("num_"), py::arg("h_feature"), py::arg("w_feature"),
        py::arg("h_mask"), py::arg("w_mask"), py::arg("half_h_mask"),
        py::arg("half_w_mask"));
  m.def("tin_shift_forward", &tin_shift_forward, "tin_shift forward",
        py::arg("input"), py::arg("shift"), py::arg("output"));
  m.def("tin_shift_backward", &tin_shift_backward, "tin_shift backward",
        py::arg("grad_output"), py::arg("shift"), py::arg("grad_input"));
  m.def("box_iou_rotated", &box_iou_rotated, "IoU for rotated boxes",
        py::arg("boxes1"), py::arg("boxes2"), py::arg("ious"),
        py::arg("mode_flag"), py::arg("aligned"));
  m.def("nms_rotated", &nms_rotated, "NMS for rotated boxes", py::arg("dets"),
        py::arg("scores"), py::arg("order"), py::arg("dets_sorted"),
        py::arg("labels"), py::arg("iou_threshold"), py::arg("multi_label"));
  m.def("ball_query_forward", &ball_query_forward, "ball_query_forward",
        py::arg("new_xyz_tensor"), py::arg("xyz_tensor"), py::arg("idx_tensor"),
        py::arg("b"), py::arg("n"), py::arg("m"), py::arg("min_radius"),
        py::arg("max_radius"), py::arg("nsample"));
  m.def("stack_ball_query_forward", &stack_ball_query_forward,
        "stack_ball_query_forward", py::arg("new_xyz_tensor"),
        py::arg("new_xyz_batch_cnt"), py::arg("xyz_tensor"),
        py::arg("xyz_batch_cnt"), py::arg("idx_tensor"), py::arg("max_radius"),
        py::arg("nsample"));
  m.def("roi_align_rotated_forward", &roi_align_rotated_forward,
        "roi_align_rotated forward", py::arg("input"), py::arg("rois"),
        py::arg("output"), py::arg("pooled_height"), py::arg("pooled_width"),
        py::arg("spatial_scale"), py::arg("sampling_ratio"), py::arg("aligned"),
        py::arg("clockwise"));
  m.def("roi_align_rotated_backward", &roi_align_rotated_backward,
        "roi_align_rotated backward", py::arg("rois"), py::arg("grad_input"),
        py::arg("grad_output"), py::arg("pooled_height"),
        py::arg("pooled_width"), py::arg("spatial_scale"),
        py::arg("sampling_ratio"), py::arg("aligned"), py::arg("clockwise"));
  m.def("dynamic_point_to_voxel_forward", &dynamic_point_to_voxel_forward,
        "dynamic_point_to_voxel_forward", py::arg("feats"), py::arg("coors"),
        py::arg("reduce_type"));
  m.def("dynamic_point_to_voxel_backward", &dynamic_point_to_voxel_backward,
        "dynamic_point_to_voxel_backward", py::arg("grad_feats"),
        py::arg("grad_reduced_feats"), py::arg("feats"),
        py::arg("reduced_feats"), py::arg("coors_idx"), py::arg("reduce_count"),
        py::arg("reduce_type"));
  m.def("hard_voxelize_forward", &hard_voxelize_forward,
        "hard_voxelize_forward", py::arg("points"), py::arg("voxel_size"),
        py::arg("coors_range"), py::arg("voxels"), py::arg("coors"),
        py::arg("num_points_per_voxel"), py::arg("voxel_num"),
        py::arg("max_points"), py::arg("max_voxels"), py::arg("NDim"),
        py::arg("deterministic"));
  m.def("dynamic_voxelize_forward", &dynamic_voxelize_forward,
        "dynamic_voxelize_forward", py::arg("points"), py::arg("voxel_size"),
        py::arg("coors_range"), py::arg("coors"), py::arg("NDim"));
  m.def("ms_deform_attn_forward", &ms_deform_attn_forward,
        "forward function of multi-scale deformable attention",
        py::arg("value"), py::arg("value_spatial_shapes"),
        py::arg("value_level_start_index"), py::arg("sampling_locations"),
        py::arg("attention_weights"), py::arg("im2col_step"));
  m.def("ms_deform_attn_backward", &ms_deform_attn_backward,
        "backward function of multi-scale deformable attention",
        py::arg("value"), py::arg("value_spatial_shapes"),
        py::arg("value_level_start_index"), py::arg("sampling_locations"),
        py::arg("attention_weights"), py::arg("grad_output"),
        py::arg("grad_value"), py::arg("grad_sampling_loc"),
        py::arg("grad_attn_weight"), py::arg("im2col_step"));
  m.def("border_align_forward", &border_align_forward,
        "forward function of border_align", py::arg("input"), py::arg("boxes"),
        py::arg("output"), py::arg("argmax_idx"), py::arg("pool_size"));
  m.def("border_align_backward", &border_align_backward,
        "backward function of border_align", py::arg("grad_output"),
        py::arg("boxes"), py::arg("argmax_idx"), py::arg("grad_input"),
        py::arg("pool_size"));
  m.def("correlation_forward", &correlation_forward, "Correlation forward",
        py::arg("input1"), py::arg("input2"), py::arg("output"), py::arg("kH"),
        py::arg("kW"), py::arg("patchH"), py::arg("patchW"), py::arg("padH"),
        py::arg("padW"), py::arg("dilationH"), py::arg("dilationW"),
        py::arg("dilation_patchH"), py::arg("dilation_patchW"), py::arg("dH"),
        py::arg("dW"));
  m.def("correlation_backward", &correlation_backward, "Correlation backward",
        py::arg("grad_output"), py::arg("input1"), py::arg("input2"),
        py::arg("grad_input1"), py::arg("grad_input2"), py::arg("kH"),
        py::arg("kW"), py::arg("patchH"), py::arg("patchW"), py::arg("padH"),
        py::arg("padW"), py::arg("dilationH"), py::arg("dilationW"),
        py::arg("dilation_patchH"), py::arg("dilation_patchW"), py::arg("dH"),
        py::arg("dW"));
  m.def("points_in_boxes_cpu_forward", &points_in_boxes_cpu_forward,
        "points_in_boxes_cpu_forward", py::arg("boxes_tensor"),
        py::arg("pts_tensor"), py::arg("pts_indices_tensor"));
  m.def("points_in_boxes_part_forward", &points_in_boxes_part_forward,
        "points_in_boxes_part_forward", py::arg("boxes_tensor"),
        py::arg("pts_tensor"), py::arg("box_idx_of_points_tensor"));
  m.def("points_in_boxes_all_forward", &points_in_boxes_all_forward,
        "points_in_boxes_all_forward", py::arg("boxes_tensor"),
        py::arg("pts_tensor"), py::arg("box_idx_of_points_tensor"));
  m.def("roiaware_pool3d_forward", &roiaware_pool3d_forward,
        "roiaware_pool3d_forward", py::arg("rois"), py::arg("pts"),
        py::arg("pts_feature"), py::arg("argmax"), py::arg("pts_idx_of_voxels"),
        py::arg("pooled_features"), py::arg("pool_method"));
  m.def("roiaware_pool3d_backward", &roiaware_pool3d_backward,
        "roiaware_pool3d_backward", py::arg("pts_idx_of_voxels"),
        py::arg("argmax"), py::arg("grad_out"), py::arg("grad_in"),
        py::arg("pool_method"));
  m.def("rotated_feature_align_forward", &rotated_feature_align_forward,
        "Feature Refine forward (CUDA)", py::arg("features"),
        py::arg("best_bboxes"), py::arg("output"), py::arg("spatial_scale"),
        py::arg("points"));
  m.def("rotated_feature_align_backward", &rotated_feature_align_backward,
        "Feature Refine backward (CUDA)", py::arg("top_grad"),
        py::arg("best_bboxes"), py::arg("bottom_grad"),
        py::arg("spatial_scale"), py::arg("points"));
  m.def("riroi_align_rotated_forward", &riroi_align_rotated_forward,
        "riroi_align_rotated forward", py::arg("features"), py::arg("rois"),
        py::arg("output"), py::arg("pooled_height"), py::arg("pooled_width"),
        py::arg("spatial_scale"), py::arg("num_samples"),
        py::arg("num_orientations"), py::arg("clockwise"));
  m.def("riroi_align_rotated_backward", &riroi_align_rotated_backward,
        "riroi_align_rotated backward", py::arg("top_grad"), py::arg("rois"),
        py::arg("bottom_grad"), py::arg("pooled_height"),
        py::arg("pooled_width"), py::arg("spatial_scale"),
        py::arg("num_samples"), py::arg("num_orientations"),
        py::arg("clockwise"));
  m.def("points_in_polygons_forward", &points_in_polygons_forward,
        "points_in_polygons_forward", py::arg("points"), py::arg("polygons"),
        py::arg("output"));
  m.def("min_area_polygons", &min_area_polygons, "min_area_polygons",
        py::arg("pointsets"), py::arg("polygons"));
  m.def("active_rotated_filter_forward", &active_rotated_filter_forward,
        "active_rotated_filter_forward", py::arg("input"), py::arg("indices"),
        py::arg("output"));
  m.def("active_rotated_filter_backward", &active_rotated_filter_backward,
        "active_rotated_filter_backward", py::arg("grad_out"),
        py::arg("indices"), py::arg("grad_in"));
  m.def("convex_iou", &convex_iou, "convex_iou", py::arg("pointsets"),
        py::arg("polygons"), py::arg("ious"));
  m.def("convex_giou", &convex_giou, "convex_giou", py::arg("pointsets"),
        py::arg("polygons"), py::arg("output"));
  m.def("diff_iou_rotated_sort_vertices_forward",
        &diff_iou_rotated_sort_vertices_forward,
        "diff_iou_rotated_sort_vertices_forward", py::arg("vertices"),
        py::arg("mask"), py::arg("num_valid"));
  m.def("chamfer_distance_forward", &chamfer_distance_forward,
        "chamfer_distance_forward", py::arg("xyz1"), py::arg("xyz2"),
        py::arg("dist1"), py::arg("dist2"), py::arg("idx1"), py::arg("idx2"));
  m.def("chamfer_distance_backward", &chamfer_distance_backward,
        "chamfer_distance_backward", py::arg("xyz1"), py::arg("xyz2"),
        py::arg("idx1"), py::arg("idx2"), py::arg("graddist1"),
        py::arg("graddist2"), py::arg("gradxyz1"), py::arg("gradxyz2"));
  m.def("prroi_pool_forward", &prroi_pool_forward, "prroi_pool forward",
        py::arg("input"), py::arg("rois"), py::arg("output"),
        py::arg("pooled_height"), py::arg("pooled_width"),
        py::arg("spatial_scale"));
  m.def("prroi_pool_backward", &prroi_pool_backward, "prroi_pool_backward",
        py::arg("grad_output"), py::arg("rois"), py::arg("grad_input"),
        py::arg("pooled_height"), py::arg("pooled_width"),
        py::arg("spatial_scale"));
  m.def("prroi_pool_coor_backward", &prroi_pool_coor_backward,
        "prroi_pool_coor_backward", py::arg("output"), py::arg("grad_output"),
        py::arg("input"), py::arg("rois"), py::arg("grad_rois"),
        py::arg("pooled_height"), py::arg("pooled_width"),
        py::arg("spatial_scale"));
  m.def("box_iou_quadri", &box_iou_quadri, "IoU for quadrilateral boxes",
        py::arg("boxes1"), py::arg("boxes2"), py::arg("ious"),
        py::arg("mode_flag"), py::arg("aligned"));
  m.def("nms_quadri", &nms_quadri, "NMS for quadrilateral boxes",
        py::arg("dets"), py::arg("scores"), py::arg("order"),
        py::arg("dets_sorted"), py::arg("iou_threshold"),
        py::arg("multi_label"));
}
