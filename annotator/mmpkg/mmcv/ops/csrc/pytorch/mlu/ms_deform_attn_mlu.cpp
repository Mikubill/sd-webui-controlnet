/*************************************************************************
 * Copyright (C) 2022 by Cambricon.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#include "pytorch_device_registry.hpp"
#include "pytorch_mlu_helper.hpp"

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

typedef enum {
  MS_DEFORM_ATTN_FORWARD_INVALID = 0, /*!< Index is invalid. */
  MS_DEFORM_ATTN_FORWARD_DEFAULT =
      1, /*!< MLUKernelMsDeformAttnForwardDefault */
  MS_DEFORM_ATTN_FORWARD_SMALL_CHANNEL =
      2, /*!< MLUKernelMsDeformAttnForwardSmallChannel */
} MsDeformAttnForwardPolicy;

void KernelMsDeformAttnForwardDefault(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const cnrtDataType_t d_type, const char* data_value_gdram,
    const char* data_spatial_shapes_gdram,
    const char* data_level_start_index_gdram,
    const char* data_sampling_loc_gdram, const char* data_attn_weight_gdram,
    const int32_t batch_size, const int32_t num_keys, const int32_t num_heads,
    const int32_t channels, const int32_t num_levels, const int32_t num_queries,
    const int32_t num_points, char* data_col_gdram);
void KernelMsDeformAttnForwardSmallChannel(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const cnrtDataType_t d_type, const char* data_value_gdram,
    const char* data_spatial_shapes_gdram,
    const char* data_level_start_index_gdram,
    const char* data_sampling_loc_gdram, const char* data_attn_weight_gdram,
    const int32_t batch_size, const int32_t num_keys, const int32_t num_heads,
    const int32_t channels, const int32_t num_levels, const int32_t num_queries,
    const int32_t num_points, char* data_col_gdram);

typedef enum {
  MS_DEFORM_ATTN_BACKWARD_DEFAULT = 0,
  MS_DEFORM_ATTN_BACKWARD_SMALL_CHANNEL = 1,
} MsDeformAttnBackwardKernelPolicy;

MsDeformAttnBackwardKernelPolicy msDeformAttnBackwardPolicyFunc(
    const int32_t channels, const int32_t num_levels, const int32_t num_points,
    const int32_t num_heads) {
  const int32_t nram_size = torch_mlu::getDeviceAttr(cnrtAttrNramSizePerMcore);
  const int num_hlp = num_heads * num_levels * num_points;
  int num_per_time_theory = (nram_size - num_levels * sizeof(float) -
                             3 * num_levels * sizeof(int32_t)) /
                            sizeof(float) / (8 * PAD_UP(channels, 32) + 28) /
                            PAD_UP((num_hlp), 32);
  if (num_per_time_theory >= 1) {
    return MS_DEFORM_ATTN_BACKWARD_SMALL_CHANNEL;
  }
  return MS_DEFORM_ATTN_BACKWARD_DEFAULT;
}

void KernelMsDeformAttnBackwardDefaultKernel(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const cnrtDataType_t d_type, const float* data_value,
    const int32_t* spatial_shapes, const int32_t* data_level_start_index,
    const float* data_sampling_loc, const float* data_attn_weight,
    const float* grad_output, const int32_t batch_size, const int32_t num_keys,
    const int32_t num_heads, const int32_t channels, const int32_t num_levels,
    const int32_t num_queries, const int32_t num_points, float* grad_value,
    float* grad_sampling_loc, float* grad_attn_weight);

void KernelMsDeformAttnBackwardSmallChannelsKernel(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const cnrtDataType_t d_type, const float* data_value,
    const int32_t* spatial_shapes, const int32_t* data_level_start_index,
    const float* data_sampling_loc, const float* data_attn_weight,
    const float* grad_output, const int32_t batch, const int32_t spatial_size,
    const int32_t num_heads, const int32_t channels, const int32_t num_levels,
    const int32_t num_query, const int32_t num_points, float* grad_value,
    float* grad_sampling_loc, float* grad_attn_weight);

// policy function
MsDeformAttnForwardPolicy msDeformAttnForwardPolicyFunc(
    cnrtDim3_t* k_dim, cnrtFunctionType_t* k_type, const int32_t batch_size,
    const int32_t num_keys, const int32_t num_heads, const int32_t channels,
    const int32_t num_levels, const int32_t num_queries,
    const int32_t num_points) {
  k_dim->x = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);
  k_dim->y =
      MIN((batch_size * num_queries * num_heads + k_dim->x - 1) / k_dim->x,
          torch_mlu::getDeviceAttr(cnrtAttrClusterCount));
  k_dim->z = 1;
#if __BANG_ARCH__ == 520
  *k_type = CNRT_FUNC_TYPE_BLOCK;
#else
  *k_type = CNRT_FUNC_TYPE_UNION1;
#endif

  int32_t nram_size = torch_mlu::getDeviceAttr(cnrtAttrNramSizePerMcore);
  if (num_levels * num_points * 3 * sizeof(int32_t) > nram_size) {
    return MS_DEFORM_ATTN_FORWARD_DEFAULT;
  } else if (channels > nram_size / 12 / sizeof(float) || channels > 96 ||
             channels < 16) {
    return MS_DEFORM_ATTN_FORWARD_DEFAULT;
  } else {
    return MS_DEFORM_ATTN_FORWARD_SMALL_CHANNEL;
  }
}

// policy function for backward
static void policyFuncBackward(const int32_t batch_size,
                               const int32_t num_queries,
                               const int32_t num_heads,
                               const int32_t num_levels,
                               cnrtFunctionType_t* k_type, cnrtDim3_t* k_dim) {
  size_t cluster_limit = torch_mlu::getDeviceAttr(cnrtAttrClusterCount);
  size_t core_limit = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);
  k_dim->x = core_limit;
  int32_t total_num = batch_size * num_queries * num_heads * num_levels;
  size_t total_num_align = CEIL_ALIGN(total_num, core_limit);
  k_dim->y = (total_num_align / core_limit) > cluster_limit
                 ? cluster_limit
                 : (total_num_align / core_limit);
  k_dim->z = 1;
  *k_type = CNRT_FUNC_TYPE_UNION1;
}

Tensor ms_deform_attn_mlu_forward(const Tensor& value,
                                  const Tensor& spatial_shapes,
                                  const Tensor& level_start_index,
                                  const Tensor& sampling_loc,
                                  const Tensor& attn_weight,
                                  const int im2col_step) {
  // check contiguous
  AT_ASSERTM(value.is_contiguous(), "value tensor has to be contiguous");
  AT_ASSERTM(spatial_shapes.is_contiguous(),
             "spatial_shapes tensor has to be contiguous");
  AT_ASSERTM(level_start_index.is_contiguous(),
             "level_start_index tensor has to be contiguous");
  AT_ASSERTM(sampling_loc.is_contiguous(),
             "sampling_loc tensor has to be contiguous");
  AT_ASSERTM(attn_weight.is_contiguous(),
             "attn_weight tensor has to be contiguous");

  // check datatype
  TORCH_CHECK((value.scalar_type() == at::kFloat),
              "value type should be Float, got ", value.scalar_type(), ".");
  TORCH_CHECK((spatial_shapes.scalar_type() == at::kInt ||
               spatial_shapes.scalar_type() == at::kLong),
              "spatial_shapes type should be Int, got ",
              spatial_shapes.scalar_type(), ".");
  TORCH_CHECK((level_start_index.scalar_type() == at::kInt ||
               level_start_index.scalar_type() == at::kLong),
              "level_start_index type should be Int, got ",
              level_start_index.scalar_type(), ".");
  TORCH_CHECK((sampling_loc.scalar_type() == at::kFloat),
              "sampling_loc type should be Float, got ",
              sampling_loc.scalar_type(), ".");
  TORCH_CHECK((attn_weight.scalar_type() == at::kFloat),
              "attn_weight type should be Float, got ",
              attn_weight.scalar_type(), ".");

  // check shape
  TORCH_CHECK(value.dim() == 4, "value should be a 4d tensor, got ",
              value.dim(), "D.");
  TORCH_CHECK(spatial_shapes.dim() == 2,
              "spatial_shapes should be a 2d tensor, got ",
              spatial_shapes.dim(), "D.");
  TORCH_CHECK(level_start_index.dim() == 1,
              "level_start_index should be a 1d tensor, got ",
              level_start_index.dim(), "D.");
  TORCH_CHECK(sampling_loc.dim() == 6,
              "sampling_loc should be a 6d tensor, got ", sampling_loc.dim(),
              "D.");
  TORCH_CHECK(attn_weight.dim() == 5, "attn_weight should be a 5d tensor, got ",
              attn_weight.dim(), "D.");

  const int batch_size = value.size(0);
  const int num_keys = value.size(1);
  const int num_heads = value.size(2);
  const int channels = value.size(3);
  const int num_levels = spatial_shapes.size(0);
  const int num_queries = sampling_loc.size(1);
  const int num_points = sampling_loc.size(4);

  TORCH_CHECK(spatial_shapes.size(1) == 2,
              "the 2nd dimensions of spatial_shapes should be 2, got ",
              spatial_shapes.size(1), ".");
  TORCH_CHECK(sampling_loc.size(5) == 2,
              "the 6th dimensions of sampling_loc should be 2, got ",
              sampling_loc.size(5), ".");
  TORCH_CHECK((sampling_loc.size(0) == batch_size),
              "the 1st dimensions of sampling_loc should be batch_size, ",
              "but now the 1st dimension of sampling_loc is ",
              sampling_loc.size(0), ", and batch_size is ", batch_size, ".");
  TORCH_CHECK((attn_weight.size(0) == batch_size),
              "the 1st dimensions of attn_weight should be batch_size, ",
              "but now the 1st dimension of attn_weight is ",
              attn_weight.size(0), ", and batch_size is ", batch_size, ".");
  TORCH_CHECK((sampling_loc.size(2) == num_heads),
              "the 3rd dimensions of sampling_loc should be num_heads, ",
              "but now the 3rd dimension of sampling_loc is ",
              sampling_loc.size(2), ", and num_heads is ", num_heads, ".");
  TORCH_CHECK((attn_weight.size(2) == num_heads),
              "the 3rd dimensions of attn_weight should be num_heads, ",
              "but now the 3rd dimension of attn_weight is ",
              attn_weight.size(2), ", and num_heads is ", num_heads, ".");
  TORCH_CHECK((level_start_index.size(0) == num_levels),
              "the 1st dimensions of level_start_index should be num_levels, ",
              "but now the 1st dimension of level_start_index is ",
              level_start_index.size(0), ", and num_levels is ", num_levels,
              ".");
  TORCH_CHECK((sampling_loc.size(3) == num_levels),
              "the 4th dimensions of sampling_loc should be num_levels, ",
              "but now the 4th dimension of sampling_loc is ",
              sampling_loc.size(3), ", and num_levels is ", num_levels, ".");
  TORCH_CHECK((attn_weight.size(3) == num_levels),
              "the 4th dimensions of attn_weight should be num_levels, ",
              "but now the 4th dimension of attn_weight is ",
              attn_weight.size(3), ", and num_levels is ", num_levels, ".");
  TORCH_CHECK((attn_weight.size(1) == num_queries),
              "the 2nd dimensions of attn_weight should be num_queries, ",
              "but now the 2nd dimension of attn_weight is ",
              attn_weight.size(1), ", and num_queries is ", num_queries, ".");
  TORCH_CHECK((attn_weight.size(4) == num_points),
              "the 5th dimensions of attn_weight should be num_points, ",
              "but now the 5th dimension of attn_weight is ",
              attn_weight.size(4), ", and num_points is ", num_points, ".");

  auto output = at::zeros({batch_size, num_queries, num_heads, channels},
                          value.options());

  // large tensor check
  const size_t max_input_size = 2147483648;
  TORCH_CHECK(value.numel() < max_input_size,
              "value element num should be less than 2^31, got ", value.numel(),
              ".");
  TORCH_CHECK(sampling_loc.numel() < max_input_size,
              "sampling_loc element num should be less than 2^31, got ",
              sampling_loc.numel(), ".");
  TORCH_CHECK(output.numel() < max_input_size,
              "output element num should be less than 2^31, got ",
              output.numel(), ".");

  // check zero element
  TORCH_CHECK(batch_size != 0, "batch_size should not be zero");
  TORCH_CHECK(num_heads != 0, "num_heads should not be zero");
  TORCH_CHECK(channels != 0, "channels should not be zero");
  TORCH_CHECK(num_queries != 0, "num_queries should not be zero");

  if (num_keys == 0 || num_levels == 0 || num_points == 0) {
    return output;
  }

  // calculate task dimension
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  MsDeformAttnForwardPolicy policy = msDeformAttnForwardPolicyFunc(
      &k_dim, &k_type, batch_size, num_keys, num_heads, channels, num_levels,
      num_queries, num_points);

  // get compute queue
  auto queue = torch_mlu::getCurQueue();

  auto spatial_shapes_ = spatial_shapes.to(at::kInt);
  auto level_start_index_ = level_start_index.to(at::kInt);

  // get ptr of tensors
  auto value_impl = torch_mlu::getMluTensorImpl(value);
  auto value_ptr = value_impl->cnnlMalloc();
  auto spatial_shapes_impl = torch_mlu::getMluTensorImpl(spatial_shapes_);
  auto spatial_shapes_ptr = spatial_shapes_impl->cnnlMalloc();
  auto level_start_index_impl = torch_mlu::getMluTensorImpl(level_start_index_);
  auto level_start_index_ptr = level_start_index_impl->cnnlMalloc();
  auto sampling_loc_impl = torch_mlu::getMluTensorImpl(sampling_loc);
  auto sampling_loc_ptr = sampling_loc_impl->cnnlMalloc();
  auto attn_weight_impl = torch_mlu::getMluTensorImpl(attn_weight);
  auto attn_weight_ptr = attn_weight_impl->cnnlMalloc();
  auto output_impl = torch_mlu::getMluTensorImpl(output);
  auto output_ptr = output_impl->cnnlMalloc();

  // get compute dtype of input
  cnrtDataType_t data_type = torch_mlu::toCnrtDtype(value.dtype());

  // launch kernel
  switch (policy) {
    default: {
      VLOG(5) << "MsDeformAttnForward Policy not supported";
    }; break;
    case MS_DEFORM_ATTN_FORWARD_DEFAULT: {
      CNLOG(INFO) << "Launch Kernel MLUKernelMsDeformAttnForwardDefault<<<"
                  << k_dim.x << ", " << k_dim.y << ", " << k_dim.z << ">>>";
      KernelMsDeformAttnForwardDefault(
          k_dim, k_type, queue, data_type, (char*)value_ptr,
          (char*)spatial_shapes_ptr, (char*)level_start_index_ptr,
          (char*)sampling_loc_ptr, (char*)attn_weight_ptr, batch_size, num_keys,
          num_heads, channels, num_levels, num_queries, num_points,
          (char*)output_ptr);
      break;
    }
    case MS_DEFORM_ATTN_FORWARD_SMALL_CHANNEL: {
      CNLOG(INFO) << "Launch Kernel MLUKernelMsDeformAttnForwardSmallChannel<<<"
                  << k_dim.x << ", " << k_dim.y << ", " << k_dim.z << ">>>";
      KernelMsDeformAttnForwardSmallChannel(
          k_dim, k_type, queue, data_type, (char*)value_ptr,
          (char*)spatial_shapes_ptr, (char*)level_start_index_ptr,
          (char*)sampling_loc_ptr, (char*)attn_weight_ptr, batch_size, num_keys,
          num_heads, channels, num_levels, num_queries, num_points,
          (char*)output_ptr);
      break;
    }
  }

  output = output.view({batch_size, num_queries, num_heads * channels});
  return output;
}

void ms_deform_attn_mlu_backward(
    const Tensor& value, const Tensor& spatial_shapes,
    const Tensor& level_start_index, const Tensor& sampling_loc,
    const Tensor& attn_weight, const Tensor& grad_output, Tensor& grad_value,
    Tensor& grad_sampling_loc, Tensor& grad_attn_weight,
    const int im2col_step) {
  // check contiguous
  AT_ASSERTM(value.is_contiguous(), "value tensor has to be contiguous");
  AT_ASSERTM(spatial_shapes.is_contiguous(),
             "spatial_shapes tensor has to be contiguous");
  AT_ASSERTM(level_start_index.is_contiguous(),
             "level_start_index tensor has to be contiguous");
  AT_ASSERTM(sampling_loc.is_contiguous(),
             "sampling_loc tensor has to be contiguous");
  AT_ASSERTM(attn_weight.is_contiguous(),
             "attn_weight tensor has to be contiguous");
  AT_ASSERTM(grad_output.is_contiguous(),
             "grad_output tensor has to be contiguous");

  // check datatype
  TORCH_CHECK((value.scalar_type() == at::kFloat),
              "value type should be Float, got ", value.scalar_type(), ".");
  TORCH_CHECK((spatial_shapes.scalar_type() == at::kInt ||
               spatial_shapes.scalar_type() == at::kLong),
              "spatial_shapes type should be Int, got ",
              spatial_shapes.scalar_type(), ".");
  TORCH_CHECK((level_start_index.scalar_type() == at::kInt ||
               level_start_index.scalar_type() == at::kLong),
              "level_start_index type should be Int, got ",
              level_start_index.scalar_type(), ".");
  TORCH_CHECK((sampling_loc.scalar_type() == at::kFloat),
              "sampling_loc type should be Float, got ",
              sampling_loc.scalar_type(), ".");
  TORCH_CHECK((attn_weight.scalar_type() == at::kFloat),
              "attn_weight type should be Float, got ",
              attn_weight.scalar_type(), ".");
  TORCH_CHECK((grad_output.scalar_type() == at::kFloat),
              "grad_output type should be Float, got ",
              grad_output.scalar_type(), ".");

  const int batch_size = value.size(0);
  const int num_keys = value.size(1);
  const int num_heads = value.size(2);
  const int channels = value.size(3);
  const int num_levels = spatial_shapes.size(0);
  const int num_queries = sampling_loc.size(1);
  const int num_points = sampling_loc.size(4);
  // Check shape.
  TORCH_CHECK(spatial_shapes.size(1) == 2,
              "the 2nd dimensions of spatial_shapes should be 2, got ",
              spatial_shapes.size(1), ".");

  TORCH_CHECK((level_start_index.size(0) == num_levels),
              "the 1st dimensions of level_start_index should be num_levels, ",
              "but now the 1st dimension of level_start_index is ",
              level_start_index.size(0), ", and num_levels is ", num_levels,
              ".");

  TORCH_CHECK((sampling_loc.size(0) == batch_size),
              "the 1st dimensions of sampling_loc should be batch_size, ",
              "but now the 1st dimension of sampling_loc is ",
              sampling_loc.size(0), ", and batch_size is ", batch_size, ".");
  TORCH_CHECK((sampling_loc.size(2) == num_heads),
              "the 3rd dimensions of sampling_loc should be num_heads, ",
              "but now the 3rd dimension of sampling_loc is ",
              sampling_loc.size(2), ", and num_heads is ", num_heads, ".");
  TORCH_CHECK((sampling_loc.size(3) == num_levels),
              "the 4th dimensions of sampling_loc should be num_levels, ",
              "but now the 4th dimension of sampling_loc is ",
              sampling_loc.size(3), ", and num_levels is ", num_levels, ".");
  TORCH_CHECK(sampling_loc.size(5) == 2,
              "the 6th dimensions of sampling_loc should be 2, got ",
              sampling_loc.size(5), ".");

  TORCH_CHECK((attn_weight.size(0) == batch_size),
              "the 1st dimensions of attn_weight should be batch_size, ",
              "but now the 1st dimension of attn_weight is ",
              attn_weight.size(0), ", and batch_size is ", batch_size, ".");
  TORCH_CHECK((attn_weight.size(1) == num_queries),
              "the 2nd dimensions of attn_weight should be num_queries, ",
              "but now the 2nd dimension of attn_weight is ",
              attn_weight.size(1), ", and num_queries is ", num_queries, ".");

  TORCH_CHECK((attn_weight.size(2) == num_heads),
              "the 3rd dimensions of attn_weight should be num_heads, ",
              "but now the 3rd dimension of attn_weight is ",
              attn_weight.size(2), ", and num_heads is ", num_heads, ".");
  TORCH_CHECK((attn_weight.size(3) == num_levels),
              "the 4th dimensions of attn_weight should be num_levels, ",
              "but now the 4th dimension of attn_weight is ",
              attn_weight.size(3), ", and num_levels is ", num_levels, ".");
  TORCH_CHECK((attn_weight.size(4) == num_points),
              "the 5th dimensions of attn_weight should be num_points, ",
              "but now the 5th dimension of attn_weight is ",
              attn_weight.size(4), ", and num_points is ", num_points, ".");

  TORCH_CHECK((grad_output.size(0) == batch_size),
              "the 1st dimensions of grad_output should be batch_size, ",
              "but now the 1st dimension of grad_output is ",
              grad_output.size(0), ", and batch_size is ", batch_size, ".");
  TORCH_CHECK((grad_output.size(1) == num_queries),
              "the 2nd dimensions of grad_output should be num_queries, ",
              "but now the 2nd dimension of grad_output is ",
              grad_output.size(1), ", and num_queries is ", num_queries, ".");
  TORCH_CHECK(
      (grad_output.size(2) == num_heads * channels),
      "the 3rd dimensions of grad_output should be num_heads * channels, ",
      "but now the 3rd dimension of grad_output is ", grad_output.size(2),
      ", and num_heads * channels is ", num_heads * channels, ".");

  // check zero element
  TORCH_CHECK(batch_size != 0, "The batch_size is zero.");
  TORCH_CHECK(channels != 0, "The channels is zero.");
  TORCH_CHECK(num_keys != 0, "The num_keys is zero.");
  TORCH_CHECK(num_heads != 0, "The num_heads is zero.");
  TORCH_CHECK(num_queries != 0, "The num_queries is zero.");
  if (num_levels == 0 || num_points == 0) {
    return;
  }

  // calculate task dimension
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFuncBackward(batch_size, num_queries, num_heads, num_levels, &k_type,
                     &k_dim);

  // get compute queue
  auto queue = torch_mlu::getCurQueue();

  // get ptr of tensors
  auto value_impl = torch_mlu::getMluTensorImpl(value);
  auto value_ptr = value_impl->cnnlMalloc();
  auto spatial_shapes_impl = torch_mlu::getMluTensorImpl(spatial_shapes);
  auto spatial_shapes_ptr = spatial_shapes_impl->cnnlMalloc();
  auto level_start_index_impl = torch_mlu::getMluTensorImpl(level_start_index);
  auto level_start_index_ptr = level_start_index_impl->cnnlMalloc();
  auto sampling_loc_impl = torch_mlu::getMluTensorImpl(sampling_loc);
  auto sampling_loc_ptr = sampling_loc_impl->cnnlMalloc();
  auto attn_weight_impl = torch_mlu::getMluTensorImpl(attn_weight);
  auto attn_weight_ptr = attn_weight_impl->cnnlMalloc();
  auto grad_output_impl = torch_mlu::getMluTensorImpl(grad_output);
  auto grad_output_ptr = grad_output_impl->cnnlMalloc();
  auto grad_value_impl = torch_mlu::getMluTensorImpl(grad_value);
  auto grad_value_ptr = grad_value_impl->cnnlMalloc();
  auto grad_sampling_loc_impl = torch_mlu::getMluTensorImpl(grad_sampling_loc);
  auto grad_sampling_loc_ptr = grad_sampling_loc_impl->cnnlMalloc();
  auto grad_attn_weight_impl = torch_mlu::getMluTensorImpl(grad_attn_weight);
  auto grad_attn_weight_ptr = grad_attn_weight_impl->cnnlMalloc();

  // get comput dtype of input
  cnrtDataType_t data_type = torch_mlu::toCnrtDtype(value.dtype());

  // launch kernel
  CNLOG(INFO) << "Launch Kernel MLUKernelMsDeformAttnBackward<<<" << k_dim.x
              << ", " << k_dim.y << ", " << k_dim.z << ">>>";
  MsDeformAttnBackwardKernelPolicy kernelPolicy =
      msDeformAttnBackwardPolicyFunc(channels, num_levels, num_points,
                                     num_heads);
  switch (kernelPolicy) {
    default: {
      VLOG(5) << "NotImplemented.";
    } break;
    case MS_DEFORM_ATTN_BACKWARD_DEFAULT: {
      KernelMsDeformAttnBackwardDefaultKernel(
          k_dim, k_type, queue, data_type, (float*)value_ptr,
          (int32_t*)spatial_shapes_ptr, (int32_t*)level_start_index_ptr,
          (float*)sampling_loc_ptr, (float*)attn_weight_ptr,
          (float*)grad_output_ptr, batch_size, num_keys, num_heads, channels,
          num_levels, num_queries, num_points, (float*)grad_value_ptr,
          (float*)grad_sampling_loc_ptr, (float*)grad_attn_weight_ptr);
    } break;
    case MS_DEFORM_ATTN_BACKWARD_SMALL_CHANNEL: {
      KernelMsDeformAttnBackwardSmallChannelsKernel(
          k_dim, k_type, queue, data_type, (float*)value_ptr,
          (int32_t*)spatial_shapes_ptr, (int32_t*)level_start_index_ptr,
          (float*)sampling_loc_ptr, (float*)attn_weight_ptr,
          (float*)grad_output_ptr, batch_size, num_keys, num_heads, channels,
          num_levels, num_queries, num_points, (float*)grad_value_ptr,
          (float*)grad_sampling_loc_ptr, (float*)grad_attn_weight_ptr);
    } break;
  }
}

Tensor ms_deform_attn_impl_forward(const Tensor& value,
                                   const Tensor& spatial_shapes,
                                   const Tensor& level_start_index,
                                   const Tensor& sampling_loc,
                                   const Tensor& attn_weight,
                                   const int im2col_step);

void ms_deform_attn_impl_backward(
    const Tensor& value, const Tensor& spatial_shapes,
    const Tensor& level_start_index, const Tensor& sampling_loc,
    const Tensor& attn_weight, const Tensor& grad_output, Tensor& grad_value,
    Tensor& grad_sampling_loc, Tensor& grad_attn_weight, const int im2col_step);

REGISTER_DEVICE_IMPL(ms_deform_attn_impl_forward, MLU,
                     ms_deform_attn_mlu_forward);
REGISTER_DEVICE_IMPL(ms_deform_attn_impl_backward, MLU,
                     ms_deform_attn_mlu_backward);
