// Copyright (c) OpenMMLab. All rights reserved
#include "trt_deform_conv.hpp"

#include <assert.h>

#include <chrono>

#include "trt_serialize.hpp"

void DeformConvForwardCUDAKernelLauncher_float(
    const float *input, const float *weight, const float *offset, float *output,
    void *workspace, int batchSize, int nInputPlane, int inputHeight,
    int inputWidth, int nOutputPlane, int kW, int kH, int dW, int dH, int padW,
    int padH, int dilationW, int dilationH, int group, int deformable_group,
    int im2col_step, cublasHandle_t cublas_handle, cudaStream_t stream);

namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"MMCVDeformConv2d"};
}  // namespace

nvinfer1::PluginFieldCollection DeformableConvPluginDynamicCreator::mFC{};
std::vector<nvinfer1::PluginField>
    DeformableConvPluginDynamicCreator::mPluginAttributes;

DeformableConvPluginDynamic::DeformableConvPluginDynamic(
    const std::string &name, const nvinfer1::Dims &stride,
    const nvinfer1::Dims &padding, const nvinfer1::Dims &dilation,
    const int deformableGroup, const int group, int im2colStep)
    : mLayerName(name),
      mStride(stride),
      mPadding(padding),
      mDilation(dilation),
      mDeformableGroup(deformableGroup),
      mGroup(group),
      mIm2colStep(im2colStep) {}

DeformableConvPluginDynamic::DeformableConvPluginDynamic(const std::string name,
                                                         const void *data,
                                                         size_t length)
    : mLayerName(name) {
  deserialize_value(&data, &length, &mStride);
  deserialize_value(&data, &length, &mPadding);
  deserialize_value(&data, &length, &mDilation);
  deserialize_value(&data, &length, &mDeformableGroup);
  deserialize_value(&data, &length, &mGroup);
  deserialize_value(&data, &length, &mIm2colStep);
}
DeformableConvPluginDynamic::~DeformableConvPluginDynamic() {}

nvinfer1::IPluginV2DynamicExt *DeformableConvPluginDynamic::clone() const {
  DeformableConvPluginDynamic *plugin =
      new DeformableConvPluginDynamic(mLayerName, mStride, mPadding, mDilation,
                                      mDeformableGroup, mGroup, mIm2colStep);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs DeformableConvPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) {
  nvinfer1::DimsExprs ret;
  ret.nbDims = 4;
  ret.d[0] = inputs[0].d[0];
  ret.d[1] = inputs[2].d[0];

  ret.d[2] = inputs[1].d[2];
  ret.d[3] = inputs[1].d[3];

  return ret;
}

bool DeformableConvPluginDynamic::supportsFormatCombination(
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

void DeformableConvPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs) {}

size_t DeformableConvPluginDynamic::getWorkspaceSize(
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

  int kW = inputs[2].dims.d[2];
  int kH = inputs[2].dims.d[3];
  int im2col_step = std::min(batch_size, mIm2colStep);

  size_t col_size =
      mmcv::getAlignedSize(nInputPlane * kW * kH * im2col_step * outputHeight *
                           outputWidth * sizeof_dtype);

  size_t out_size = 0;
  if (im2col_step != 1)
    out_size = mmcv::getAlignedSize(batch_size * nOutputPlane * outputHeight *
                                    outputWidth * sizeof_dtype);

  return col_size + out_size;
}

int DeformableConvPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *inputDesc,
    const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
    void *const *outputs, void *workSpace, cudaStream_t stream) {
  int batch_size = inputDesc[0].dims.d[0];
  int inputChannel = inputDesc[0].dims.d[1];
  int inputHeight = inputDesc[0].dims.d[2];
  int inputWidth = inputDesc[0].dims.d[3];
  int outputChannel = outputDesc[0].dims.d[1];
  int kernelHeight = inputDesc[2].dims.d[2];
  int kernelWidth = inputDesc[2].dims.d[3];

  const void *x = inputs[0];
  const void *offset = inputs[1];
  const void *weight = inputs[2];
  void *output = outputs[0];
  int im2col_step = std::min(batch_size, mIm2colStep);

  // TODO: add fp16 support
  auto data_type = inputDesc[0].type;
  switch (data_type) {
    case nvinfer1::DataType::kFLOAT:
      DeformConvForwardCUDAKernelLauncher_float(
          (float *)x, (float *)weight, (float *)offset, (float *)output,
          workSpace, batch_size, inputChannel, inputHeight, inputWidth,
          outputChannel, kernelWidth, kernelHeight, mStride.d[0], mStride.d[1],
          mPadding.d[0], mPadding.d[1], mDilation.d[0], mDilation.d[1], mGroup,
          mDeformableGroup, im2col_step, m_cublas_handle, stream);
      break;
    default:
      return 1;
      break;
  }

  return 0;
}

nvinfer1::DataType DeformableConvPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType *inputTypes, int nbInputs) const {
  return inputTypes[0];
}

// IPluginV2 Methods
const char *DeformableConvPluginDynamic::getPluginType() const {
  return PLUGIN_NAME;
}

const char *DeformableConvPluginDynamic::getPluginVersion() const {
  return PLUGIN_VERSION;
}

int DeformableConvPluginDynamic::getNbOutputs() const { return 1; }

int DeformableConvPluginDynamic::initialize() { return 0; }

void DeformableConvPluginDynamic::terminate() {}

size_t DeformableConvPluginDynamic::getSerializationSize() const {
  return sizeof(mStride) + sizeof(mPadding) + sizeof(mDilation) +
         sizeof(mDeformableGroup) + sizeof(mGroup) + sizeof(mIm2colStep);
}

void DeformableConvPluginDynamic::serialize(void *buffer) const {
  serialize_value(&buffer, mStride);
  serialize_value(&buffer, mPadding);
  serialize_value(&buffer, mDilation);
  serialize_value(&buffer, mDeformableGroup);
  serialize_value(&buffer, mGroup);
  serialize_value(&buffer, mIm2colStep);
}

void DeformableConvPluginDynamic::destroy() {
  // This gets called when the network containing plugin is destroyed
  delete this;
}

void DeformableConvPluginDynamic::attachToContext(
    cudnnContext *cudnnContext, cublasContext *cublasContext,
    nvinfer1::IGpuAllocator *gpuAllocator) {
  m_cublas_handle = cublasContext;
}

void DeformableConvPluginDynamic::detachFromContext() {}

void DeformableConvPluginDynamic::setPluginNamespace(const char *libNamespace) {
  mNamespace = libNamespace;
}

const char *DeformableConvPluginDynamic::getPluginNamespace() const {
  return mNamespace.c_str();
}

////////////////////// creator /////////////////////////////

DeformableConvPluginDynamicCreator::DeformableConvPluginDynamicCreator() {
  mPluginAttributes.emplace_back(nvinfer1::PluginField("stride"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("padding"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("dilation"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("groups"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("deform_groups"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("bias"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("im2col_step"));
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *DeformableConvPluginDynamicCreator::getPluginName() const {
  return PLUGIN_NAME;
}

const char *DeformableConvPluginDynamicCreator::getPluginVersion() const {
  return PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection *
DeformableConvPluginDynamicCreator::getFieldNames() {
  return &mFC;
}

nvinfer1::IPluginV2 *DeformableConvPluginDynamicCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) {
  nvinfer1::Dims stride{2, {1, 1}};
  nvinfer1::Dims padding{2, {0, 0}};
  nvinfer1::Dims dilation{2, {1, 1}};
  int deformableGroup = 1;
  int group = 1;
  int im2col_step = 32;

  for (int i = 0; i < fc->nbFields; i++) {
    if (fc->fields[i].data == nullptr) {
      continue;
    }
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("stride") == 0) {
      stride.nbDims = 2;
      stride.d[0] = static_cast<const int *>(fc->fields[i].data)[0];
      if (fc->fields[i].length == 1) {
        stride.d[1] = stride.d[0];
      } else {
        stride.d[1] = static_cast<const int *>(fc->fields[i].data)[1];
      }
    }

    if (field_name.compare("padding") == 0) {
      padding.nbDims = 2;
      padding.d[0] = static_cast<const int *>(fc->fields[i].data)[0];
      if (fc->fields[i].length == 1) {
        padding.d[1] = padding.d[0];
      } else {
        padding.d[1] = static_cast<const int *>(fc->fields[i].data)[1];
      }
    }

    if (field_name.compare("dilation") == 0) {
      dilation.nbDims = 2;
      dilation.d[0] = static_cast<const int *>(fc->fields[i].data)[0];
      if (fc->fields[i].length == 1) {
        dilation.d[1] = dilation.d[0];
      } else {
        dilation.d[1] = static_cast<const int *>(fc->fields[i].data)[1];
      }
    }

    if (field_name.compare("deform_groups") == 0) {
      deformableGroup = static_cast<const int *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("groups") == 0) {
      group = static_cast<const int *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("im2col_step") == 0) {
      im2col_step = static_cast<const int *>(fc->fields[i].data)[0];
    }
  }

  DeformableConvPluginDynamic *plugin = new DeformableConvPluginDynamic(
      name, stride, padding, dilation, deformableGroup, group, im2col_step);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::IPluginV2 *DeformableConvPluginDynamicCreator::deserializePlugin(
    const char *name, const void *serialData, size_t serialLength) {
  auto plugin = new DeformableConvPluginDynamic(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

void DeformableConvPluginDynamicCreator::setPluginNamespace(
    const char *libNamespace) {
  mNamespace = libNamespace;
}

const char *DeformableConvPluginDynamicCreator::getPluginNamespace() const {
  return mNamespace.c_str();
}
