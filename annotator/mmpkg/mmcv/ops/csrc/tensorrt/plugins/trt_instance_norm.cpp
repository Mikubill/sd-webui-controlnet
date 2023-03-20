// Copyright (c) OpenMMLab. All rights reserved
// Modified from:
// https://github.com/NVIDIA/TensorRT/blob/master/plugin/instanceNormalizationPlugin/instanceNormalizationPlugin.cpp

#include "trt_instance_norm.hpp"

#include <cuda_fp16.h>

#include <stdexcept>

#include "trt_serialize.hpp"

using namespace nvinfer1;

cudnnStatus_t convert_trt2cudnn_dtype(nvinfer1::DataType trt_dtype,
                                      cudnnDataType_t* cudnn_dtype) {
  switch (trt_dtype) {
    case nvinfer1::DataType::kFLOAT:
      *cudnn_dtype = CUDNN_DATA_FLOAT;
      break;
    case nvinfer1::DataType::kHALF:
      *cudnn_dtype = CUDNN_DATA_HALF;
      break;
    default:
      return CUDNN_STATUS_BAD_PARAM;
  }
  return CUDNN_STATUS_SUCCESS;
}

namespace {
constexpr const char* PLUGIN_VERSION{"1"};
constexpr const char* PLUGIN_NAME{"MMCVInstanceNormalization"};
}  // namespace

PluginFieldCollection InstanceNormalizationDynamicCreator::mFC{};
std::vector<PluginField> InstanceNormalizationDynamicCreator::mPluginAttributes;

InstanceNormalizationDynamic::InstanceNormalizationDynamic(
    const std::string& name, float epsilon)
    : mLayerName(name), mEpsilon(epsilon) {}

InstanceNormalizationDynamic::InstanceNormalizationDynamic(
    const std::string& name, void const* serialData, size_t serialLength)
    : mLayerName(name) {
  deserialize_value(&serialData, &serialLength, &mEpsilon);
}

InstanceNormalizationDynamic::~InstanceNormalizationDynamic() {}

// InstanceNormalizationDynamic returns one output.
int InstanceNormalizationDynamic::getNbOutputs() const { return 1; }

DimsExprs InstanceNormalizationDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) {
  nvinfer1::DimsExprs output(inputs[0]);
  return output;
}

int InstanceNormalizationDynamic::initialize() { return 0; }

void InstanceNormalizationDynamic::terminate() {}

size_t InstanceNormalizationDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const {
  int n = inputs[0].dims.d[0];
  int c = inputs[0].dims.d[1];
  int elem_size = mmcv::getElementSize(inputs[1].type);
  return mmcv::getAlignedSize(n * c * elem_size) * 2;
}

int InstanceNormalizationDynamic::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) {
  nvinfer1::Dims input_dims = inputDesc[0].dims;
  int n = input_dims.d[0];
  int c = input_dims.d[1];
  int h = input_dims.d[2];
  int w = input_dims.nbDims > 3 ? input_dims.d[3] : 1;
  int elem_size = mmcv::getElementSize(inputDesc[1].type);

  void* n_scales = (void*)workspace;
  void* n_bias = (void*)(workspace + mmcv::getAlignedSize(n * c * elem_size));

  const void* scales = (const void*)inputs[1];
  const void* bias = (const void*)inputs[2];

  for (int i = 0; i < n; ++i) {
    cudaMemcpyAsync(n_scales + i * c * elem_size, scales, c * elem_size,
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(n_bias + i * c * elem_size, bias, c * elem_size,
                    cudaMemcpyDeviceToDevice, stream);
  }

  cudnnSetTensor4dDescriptor(_b_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1,
                             n * c, 1, 1);
  cudnnDataType_t cudnn_dtype{};
  convert_trt2cudnn_dtype(inputDesc[0].type, &cudnn_dtype);
  cudnnSetTensor4dDescriptor(_x_desc, CUDNN_TENSOR_NCHW, cudnn_dtype, 1, n * c,
                             h, w);
  cudnnSetTensor4dDescriptor(_y_desc, CUDNN_TENSOR_NCHW, cudnn_dtype, 1, n * c,
                             h, w);
  float alpha = 1;
  float beta = 0;
  void const* x_ptr = inputs[0];
  void* y_ptr = outputs[0];
  cudnnSetStream(_cudnn_handle, stream);
  // Note: Use of CUDNN_BATCHNORM_SPATIAL_PERSISTENT can cause numerical
  //       overflows (NaNs) for fp32 data in some circumstances. The lower-
  //       performance CUDNN_BATCHNORM_SPATIAL should be used if this is not
  //       acceptable.
  cudnnBatchNormalizationForwardTraining(
      _cudnn_handle, CUDNN_BATCHNORM_SPATIAL_PERSISTENT, &alpha, &beta, _x_desc,
      x_ptr, _y_desc, y_ptr, _b_desc, n_scales, n_bias, 1., nullptr, nullptr,
      mEpsilon, nullptr, nullptr);
  return 0;
}

size_t InstanceNormalizationDynamic::getSerializationSize() const {
  return serialized_size(mEpsilon);
}

void InstanceNormalizationDynamic::serialize(void* buffer) const {
  serialize_value(&buffer, mEpsilon);
}

bool InstanceNormalizationDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
    int nbOutputs) {
  return ((inOut[pos].type == nvinfer1::DataType::kFLOAT ||
           inOut[pos].type == nvinfer1::DataType::kHALF) &&
          inOut[pos].format == nvinfer1::PluginFormat::kLINEAR &&
          inOut[pos].type == inOut[0].type);
}

const char* InstanceNormalizationDynamic::getPluginType() const {
  return PLUGIN_NAME;
}

const char* InstanceNormalizationDynamic::getPluginVersion() const {
  return PLUGIN_VERSION;
}

void InstanceNormalizationDynamic::destroy() { delete this; }

IPluginV2DynamicExt* InstanceNormalizationDynamic::clone() const {
  auto* plugin = new InstanceNormalizationDynamic{mLayerName, mEpsilon};
  plugin->setPluginNamespace(mPluginNamespace.c_str());
  return plugin;
}

// Set plugin namespace
void InstanceNormalizationDynamic::setPluginNamespace(
    const char* pluginNamespace) {
  mPluginNamespace = pluginNamespace;
}

const char* InstanceNormalizationDynamic::getPluginNamespace() const {
  return mPluginNamespace.c_str();
}

nvinfer1::DataType InstanceNormalizationDynamic::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const {
  return inputTypes[0];
}

// Attach the plugin object to an execution context and grant the plugin the
// access to some context resource.
void InstanceNormalizationDynamic::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext,
    IGpuAllocator* gpuAllocator) {
  _cudnn_handle = cudnnContext;
  cudnnCreateTensorDescriptor(&_b_desc);
  cudnnCreateTensorDescriptor(&_x_desc);
  cudnnCreateTensorDescriptor(&_y_desc);
}

// Detach the plugin object from its execution context.
void InstanceNormalizationDynamic::detachFromContext() {
  cudnnDestroyTensorDescriptor(_y_desc);
  cudnnDestroyTensorDescriptor(_x_desc);
  cudnnDestroyTensorDescriptor(_b_desc);
}

void InstanceNormalizationDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) {}

// InstanceNormalizationDynamicCreator methods
InstanceNormalizationDynamicCreator::InstanceNormalizationDynamicCreator() {
  mPluginAttributes.clear();
  mPluginAttributes.emplace_back(
      PluginField("epsilon", nullptr, PluginFieldType::kFLOAT32, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* InstanceNormalizationDynamicCreator::getPluginName() const {
  return PLUGIN_NAME;
}

const char* InstanceNormalizationDynamicCreator::getPluginVersion() const {
  return PLUGIN_VERSION;
}

const PluginFieldCollection*
InstanceNormalizationDynamicCreator::getFieldNames() {
  return &mFC;
}

IPluginV2DynamicExt* InstanceNormalizationDynamicCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) {
  float epsilon = 1e-5;
  const PluginField* fields = fc->fields;
  for (int i = 0; i < fc->nbFields; ++i) {
    const char* attrName = fields[i].name;
    if (!strcmp(attrName, "epsilon")) {
      epsilon = *(static_cast<const float*>(fields[i].data));
    }
  }

  InstanceNormalizationDynamic* obj =
      new InstanceNormalizationDynamic(name, epsilon);
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}

IPluginV2DynamicExt* InstanceNormalizationDynamicCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) {
  InstanceNormalizationDynamic* obj =
      new InstanceNormalizationDynamic{name, serialData, serialLength};
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}

void InstanceNormalizationDynamicCreator::setPluginNamespace(
    const char* libNamespace) {
  mNamespace = libNamespace;
}

const char* InstanceNormalizationDynamicCreator::getPluginNamespace() const {
  return mNamespace.c_str();
}
