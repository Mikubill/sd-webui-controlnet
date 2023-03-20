// Modified from
// https://github.com/CVMI-Lab/PAConv/tree/main/scene_seg/lib/paconv_lib/src/gpu
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void assign_score_withk_forward_impl(int B, int N0, int N1, int M, int K, int O,
                                     int aggregate, const Tensor& points,
                                     const Tensor& centers,
                                     const Tensor& scores,
                                     const Tensor& knn_idx, Tensor& output) {
  DISPATCH_DEVICE_IMPL(assign_score_withk_forward_impl, B, N0, N1, M, K, O,
                       aggregate, points, centers, scores, knn_idx, output);
}

void assign_score_withk_backward_impl(
    int B, int N0, int N1, int M, int K, int O, int aggregate,
    const Tensor& grad_out, const Tensor& points, const Tensor& centers,
    const Tensor& scores, const Tensor& knn_idx, Tensor& grad_points,
    Tensor& grad_centers, Tensor& grad_scores) {
  DISPATCH_DEVICE_IMPL(assign_score_withk_backward_impl, B, N0, N1, M, K, O,
                       aggregate, grad_out, points, centers, scores, knn_idx,
                       grad_points, grad_centers, grad_scores);
}

void assign_score_withk_forward(const Tensor& points, const Tensor& centers,
                                const Tensor& scores, const Tensor& knn_idx,
                                Tensor& output, int B, int N0, int N1, int M,
                                int K, int O, int aggregate) {
  assign_score_withk_forward_impl(B, N0, N1, M, K, O, aggregate, points,
                                  centers, scores, knn_idx, output);
}

void assign_score_withk_backward(const Tensor& grad_out, const Tensor& points,
                                 const Tensor& centers, const Tensor& scores,
                                 const Tensor& knn_idx, Tensor& grad_points,
                                 Tensor& grad_centers, Tensor& grad_scores,
                                 int B, int N0, int N1, int M, int K, int O,
                                 int aggregate) {
  assign_score_withk_backward_impl(B, N0, N1, M, K, O, aggregate, grad_out,
                                   points, centers, scores, knn_idx,
                                   grad_points, grad_centers, grad_scores);
}
