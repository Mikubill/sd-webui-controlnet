// Copyright (c) OpenMMLab. All rights reserved
#include "trt_corner_pool.hpp"

#include <assert.h>

#include "trt_serialize.hpp"

void CornerPoolForwardLauncher_float(const float *input, float *output,
                                     const int batch_size, const int channels,
                                     const int height, const int width,
                                     const int pool_type, cudaStream_t stream);

namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *CORNER_POOL_PLUGIN_NAME{"MMCVCornerPool"};
}  // namespace

CornerPoolPluginDynamic::CornerPoolPluginDynamic(const std::string &name,
                                                 TRT_CORNER_POOL_TYPE poolType)
    : mLayerName(name), mPoolType(poolType) {}

CornerPoolPluginDynamic::CornerPoolPluginDynamic(const std::string name,
                                                 const void *data,
                                                 size_t length)
    : mLayerName(name) {
  deserialize_value(&data, &length, &mPoolType);
}

CornerPoolPluginDynamic::~CornerPoolPluginDynamic() {}

nvinfer1::IPluginV2DynamicExt *CornerPoolPluginDynamic::clone() const {
  CornerPoolPluginDynamic *plugin =
      new CornerPoolPluginDynamic(mLayerName, mPoolType);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs CornerPoolPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) {
  return inputs[0];
}

bool CornerPoolPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs,
    int nbOutputs) {
  switch (pos) {
    // input[0]
    case 0:
      return inOut[pos].type == nvinfer1::DataType::kFLOAT &&
             inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
    // output[0]
    case 1:
      return inOut[pos].type == inOut[0].type &&
             inOut[pos].format == inOut[0].format;
    default:
      return false;
  }
}

void CornerPoolPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs) {}

size_t CornerPoolPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const {
  int sizeof_dtype = mmcv::getElementSize(outputs[0].type);
}

int CornerPoolPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *inputDesc,
    const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
    void *const *outputs, void *workSpace, cudaStream_t stream) {
  const void *input = inputs[0];
  void *output_value = outputs[0];

  const int batch_size = inputDesc[0].dims.d[0];
  const int channels = inputDesc[0].dims.d[1];
  const int height = inputDesc[0].dims.d[2];
  const int width = inputDesc[0].dims.d[3];

  CornerPoolForwardLauncher_float((float *)input, (float *)output_value,
                                  batch_size, channels, height, width,
                                  int(mPoolType), stream);

  return 0;
}

nvinfer1::DataType CornerPoolPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType *inputTypes, int nbInputs) const {
  return inputTypes[0];
}

// IPluginV2 Methods
const char *CornerPoolPluginDynamic::getPluginType() const {
  switch (mPoolType) {
    case TRT_CORNER_POOL_TYPE::TRT_TOP_POOL:
    case TRT_CORNER_POOL_TYPE::TRT_BOTTOM_POOL:
    case TRT_CORNER_POOL_TYPE::TRT_LEFT_POOL:
    case TRT_CORNER_POOL_TYPE::TRT_RIGHT_POOL:
      return CORNER_POOL_PLUGIN_NAME;

    default:
      return "UnknownpoolType";
  }
}

const char *CornerPoolPluginDynamic::getPluginVersion() const {
  return PLUGIN_VERSION;
}

int CornerPoolPluginDynamic::getNbOutputs() const { return 1; }

int CornerPoolPluginDynamic::initialize() { return 0; }

void CornerPoolPluginDynamic::terminate() {}

size_t CornerPoolPluginDynamic::getSerializationSize() const {
  return sizeof(mPoolType);
}

void CornerPoolPluginDynamic::serialize(void *buffer) const {
  serialize_value(&buffer, mPoolType);
}

void CornerPoolPluginDynamic::destroy() {
  // This gets called when the network containing plugin is destroyed
  delete this;
}

void CornerPoolPluginDynamic::setPluginNamespace(const char *libNamespace) {
  mNamespace = libNamespace;
}

const char *CornerPoolPluginDynamic::getPluginNamespace() const {
  return mNamespace.c_str();
}

CornerPoolPluginDynamicCreator::CornerPoolPluginDynamicCreator() {
  mPluginAttributes.clear();
  mPluginAttributes.emplace_back(nvinfer1::PluginField("mode"));
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *CornerPoolPluginDynamicCreator::getPluginName() const {
  return CORNER_POOL_PLUGIN_NAME;
}

const char *CornerPoolPluginDynamicCreator::getPluginVersion() const {
  return PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection *
CornerPoolPluginDynamicCreator::getFieldNames() {
  return &mFC;
}

nvinfer1::IPluginV2 *CornerPoolPluginDynamicCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) {
  TRT_CORNER_POOL_TYPE poolType;
  int poolMode = -1;

  for (int i = 0; i < fc->nbFields; i++) {
    if (fc->fields[i].data == nullptr) {
      continue;
    }
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("mode") == 0) {
      poolMode = static_cast<const int *>(fc->fields[i].data)[0];
    }
  }

  assert(poolMode >= 0 && poolMode <= 3);
  switch (poolMode) {
    case 0:
      poolType = TRT_CORNER_POOL_TYPE::TRT_TOP_POOL;
      break;
    case 1:
      poolType = TRT_CORNER_POOL_TYPE::TRT_BOTTOM_POOL;
      break;
    case 2:
      poolType = TRT_CORNER_POOL_TYPE::TRT_LEFT_POOL;
      break;
    case 3:
      poolType = TRT_CORNER_POOL_TYPE::TRT_RIGHT_POOL;
      break;

    default:
      break;
  }

  CornerPoolPluginDynamic *plugin = new CornerPoolPluginDynamic(name, poolType);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::IPluginV2 *CornerPoolPluginDynamicCreator::deserializePlugin(
    const char *name, const void *serialData, size_t serialLength) {
  // This object will be deleted when the network is destroyed, which will
  // call FCPluginDynamic::destroy()
  auto plugin = new CornerPoolPluginDynamic(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

void CornerPoolPluginDynamicCreator::setPluginNamespace(
    const char *libNamespace) {
  mNamespace = libNamespace;
}

const char *CornerPoolPluginDynamicCreator::getPluginNamespace() const {
  return mNamespace.c_str();
}
