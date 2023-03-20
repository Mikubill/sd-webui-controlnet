// Copyright (c) OpenMMLab. All rights reserved
#include "trt_modulated_deform_conv.hpp"

#include <assert.h>

#include <chrono>

#include "trt_serialize.hpp"

void ModulatedDeformConvForwardCUDAKernelLauncher_float(
    const float *input, const float *weight, const float *bias,
    const float *offset, const float *mask, float *output, void *workspace,
    int batch, int channels, int height, int width, int channels_out,
    int kernel_w, int kernel_h, int stride_w, int stride_h, int pad_w,
    int pad_h, int dilation_w, int dilation_h, int group, int deformable_group,
    int im2col_step, cublasHandle_t cublas_handle, cudaStream_t stream);

namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"MMCVModulatedDeformConv2d"};
}  // namespace

nvinfer1::PluginFieldCollection
    ModulatedDeformableConvPluginDynamicCreator::mFC{};
std::vector<nvinfer1::PluginField>
    ModulatedDeformableConvPluginDynamicCreator::mPluginAttributes;

ModulatedDeformableConvPluginDynamic::ModulatedDeformableConvPluginDynamic(
    const std::string &name, const nvinfer1::Dims stride,
    const nvinfer1::Dims padding, const nvinfer1::Dims dilation,
    const int deformableGroup, const int group)
    : mLayerName(name),
      mStride(stride),
      mPadding(padding),
      mDilation(dilation),
      mDeformableGroup(deformableGroup),
      mGroup(group) {
  mWithBias = false;
}

ModulatedDeformableConvPluginDynamic::ModulatedDeformableConvPluginDynamic(
    const std::string name, const void *data, size_t length)
    : mLayerName(name) {
  deserialize_value(&data, &length, &mStride);
  deserialize_value(&data, &length, &mPadding);
  deserialize_value(&data, &length, &mDilation);
  deserialize_value(&data, &length, &mDeformableGroup);
  deserialize_value(&data, &length, &mGroup);
  mWithBias = false;
}
ModulatedDeformableConvPluginDynamic::~ModulatedDeformableConvPluginDynamic() {}

nvinfer1::IPluginV2DynamicExt *ModulatedDeformableConvPluginDynamic::clone()
    const {
  ModulatedDeformableConvPluginDynamic *plugin =
      new ModulatedDeformableConvPluginDynamic(
          mLayerName, mStride, mPadding, mDilation, mDeformableGroup, mGroup);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs ModulatedDeformableConvPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) {
  nvinfer1::DimsExprs ret;
  ret.nbDims = 4;
  ret.d[0] = inputs[0].d[0];
  ret.d[1] = inputs[3].d[0];

  ret.d[2] = inputs[1].d[2];
  ret.d[3] = inputs[1].d[3];

  return ret;
}

bool ModulatedDeformableConvPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs,
    int nbOutputs) {
  if (pos == 0) {
    return (inOut[pos].type == nvinfer1::DataType::kFLOAT &&
            inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);

  } else {
    return inOut[pos].type == inOut[0].type &&
           inOut[pos].format == inOut[0].format;
  }
}

void ModulatedDeformableConvPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs) {
  if (nbInputs == 5) {
    mWithBias = true;
  }
}

size_t ModulatedDeformableConvPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const {
  int sizeof_dtype = mmcv::getElementSize(outputs[0].type);

  int batch_size = inputs[0].dims.d[0];
  int nInputPlane = inputs[0].dims.d[1];
  int inputHeight = inputs[0].dims.d[2];
  int inputWidth = inputs[0].dims.d[3];

  int nOutputPlane = outputs[0].dims.d[1];
  int outputHeight = outputs[0].dims.d[2];
  int outputWidth = outputs[0].dims.d[3];

  int kW = inputs[3].dims.d[2];
  int kH = inputs[3].dims.d[3];
  int im2col_step = std::min(32, batch_size);

  size_t col_size = mmcv::getAlignedSize(nInputPlane * kW * kH * outputHeight *
                                         outputWidth * sizeof_dtype);

  return col_size;
}

int ModulatedDeformableConvPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *inputDesc,
    const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
    void *const *outputs, void *workSpace, cudaStream_t stream) {
  int batch = inputDesc[0].dims.d[0];
  int channels = inputDesc[0].dims.d[1];
  int height = inputDesc[0].dims.d[2];
  int width = inputDesc[0].dims.d[3];
  int channels_out = outputDesc[0].dims.d[1];
  int kernel_h = inputDesc[3].dims.d[2];
  int kernel_w = inputDesc[3].dims.d[3];

  const void *x = inputs[0];
  const void *offset = inputs[1];
  const void *mask = inputs[2];
  const void *weight = inputs[3];
  const void *bias = mWithBias ? inputs[4] : nullptr;
  void *output = outputs[0];
  int im2col_step = std::min(batch, 32);

  // TODO: add fp16 support
  auto data_type = inputDesc[0].type;
  switch (data_type) {
    case nvinfer1::DataType::kFLOAT:
      ModulatedDeformConvForwardCUDAKernelLauncher_float(
          (float *)x, (float *)weight, (float *)bias, (float *)offset,
          (float *)mask, (float *)output, workSpace, batch, channels, height,
          width, channels_out, kernel_w, kernel_h, mStride.d[0], mStride.d[1],
          mPadding.d[0], mPadding.d[1], mDilation.d[0], mDilation.d[1], mGroup,
          mDeformableGroup, im2col_step, m_cublas_handle, stream);
      break;
    default:
      return 1;
      break;
  }

  return 0;
}

nvinfer1::DataType ModulatedDeformableConvPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType *inputTypes, int nbInputs) const {
  return inputTypes[0];
}

// IPluginV2 Methods
const char *ModulatedDeformableConvPluginDynamic::getPluginType() const {
  return PLUGIN_NAME;
}

const char *ModulatedDeformableConvPluginDynamic::getPluginVersion() const {
  return PLUGIN_VERSION;
}

int ModulatedDeformableConvPluginDynamic::getNbOutputs() const { return 1; }

int ModulatedDeformableConvPluginDynamic::initialize() { return 0; }

void ModulatedDeformableConvPluginDynamic::terminate() {}

size_t ModulatedDeformableConvPluginDynamic::getSerializationSize() const {
  return sizeof(mStride) + sizeof(mPadding) + sizeof(mDilation) +
         sizeof(mDeformableGroup) + sizeof(mGroup);
}

void ModulatedDeformableConvPluginDynamic::serialize(void *buffer) const {
  serialize_value(&buffer, mStride);
  serialize_value(&buffer, mPadding);
  serialize_value(&buffer, mDilation);
  serialize_value(&buffer, mDeformableGroup);
  serialize_value(&buffer, mGroup);
}

void ModulatedDeformableConvPluginDynamic::destroy() {
  // This gets called when the network containing plugin is destroyed
  delete this;
}

void ModulatedDeformableConvPluginDynamic::attachToContext(
    cudnnContext *cudnnContext, cublasContext *cublasContext,
    nvinfer1::IGpuAllocator *gpuAllocator) {
  m_cublas_handle = cublasContext;
}

void ModulatedDeformableConvPluginDynamic::detachFromContext() {}

void ModulatedDeformableConvPluginDynamic::setPluginNamespace(
    const char *libNamespace) {
  mNamespace = libNamespace;
}

const char *ModulatedDeformableConvPluginDynamic::getPluginNamespace() const {
  return mNamespace.c_str();
}

////////////////////// creator /////////////////////////////

ModulatedDeformableConvPluginDynamicCreator::
    ModulatedDeformableConvPluginDynamicCreator() {
  mPluginAttributes.emplace_back(nvinfer1::PluginField("stride"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("padding"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("dilation"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("groups"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("deform_groups"));
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *ModulatedDeformableConvPluginDynamicCreator::getPluginName() const {
  return PLUGIN_NAME;
}

const char *ModulatedDeformableConvPluginDynamicCreator::getPluginVersion()
    const {
  return PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection *
ModulatedDeformableConvPluginDynamicCreator::getFieldNames() {
  return &mFC;
}

nvinfer1::IPluginV2 *ModulatedDeformableConvPluginDynamicCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) {
  nvinfer1::Dims stride{2, {1, 1}};
  nvinfer1::Dims padding{2, {0, 0}};
  nvinfer1::Dims dilation{2, {1, 1}};
  int deformableGroup = 1;
  int group = 1;

  for (int i = 0; i < fc->nbFields; i++) {
    if (fc->fields[i].data == nullptr) {
      continue;
    }
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("deform_groups") == 0) {
      deformableGroup = static_cast<const int *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("groups") == 0) {
      group = static_cast<const int *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("stride") == 0) {
      stride.nbDims = 2;
      stride.d[0] = static_cast<const int *>(fc->fields[i].data)[0];
      stride.d[1] = static_cast<const int *>(fc->fields[i].data)[1];
    }

    if (field_name.compare("padding") == 0) {
      padding.nbDims = 2;
      padding.d[0] = static_cast<const int *>(fc->fields[i].data)[0];
      padding.d[1] = static_cast<const int *>(fc->fields[i].data)[1];
    }

    if (field_name.compare("dilation") == 0) {
      dilation.nbDims = 2;
      dilation.d[0] = static_cast<const int *>(fc->fields[i].data)[0];
      dilation.d[1] = static_cast<const int *>(fc->fields[i].data)[1];
    }
  }

  ModulatedDeformableConvPluginDynamic *plugin =
      new ModulatedDeformableConvPluginDynamic(name, stride, padding, dilation,
                                               deformableGroup, group);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::IPluginV2 *
ModulatedDeformableConvPluginDynamicCreator::deserializePlugin(
    const char *name, const void *serialData, size_t serialLength) {
  auto plugin =
      new ModulatedDeformableConvPluginDynamic(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

void ModulatedDeformableConvPluginDynamicCreator::setPluginNamespace(
    const char *libNamespace) {
  mNamespace = libNamespace;
}

const char *ModulatedDeformableConvPluginDynamicCreator::getPluginNamespace()
    const {
  return mNamespace.c_str();
}
