// Copyright (c) OpenMMLab. All rights reserved
#include "trt_cummaxmin.hpp"

#include <assert.h>

#include "trt_serialize.hpp"

void CumMaxMinForwardLauncher_float(const float *input, float *output_value,
                                    int *output_index, const int *dims,
                                    int nbDims, int cum_dim, int cum_type,
                                    cudaStream_t stream);

void CumMaxMinForwardLauncher_int32(const int *input, int *output_value,
                                    int *output_index, const int *dims,
                                    int nbDims, int cum_dim, int cum_type,
                                    cudaStream_t stream);

namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *CUMMAXMIN_PLUGIN_NAME{"cummaxmin"};
static const char *CUMMAX_PLUGIN_NAME{"cummax"};
static const char *CUMMIN_PLUGIN_NAME{"cummin"};
}  // namespace

CumMaxMinPluginDynamic::CumMaxMinPluginDynamic(const std::string &name, int dim,
                                               TRT_CUMCMPTYPE cumType)
    : mLayerName(name), mDim(dim), mCumType(cumType) {}

CumMaxMinPluginDynamic::CumMaxMinPluginDynamic(const std::string name,
                                               const void *data, size_t length)
    : mLayerName(name) {
  deserialize_value(&data, &length, &mDim);
  deserialize_value(&data, &length, &mCumType);
}

CumMaxMinPluginDynamic::~CumMaxMinPluginDynamic() {}

nvinfer1::IPluginV2DynamicExt *CumMaxMinPluginDynamic::clone() const {
  CumMaxMinPluginDynamic *plugin =
      new CumMaxMinPluginDynamic(mLayerName, mDim, mCumType);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs CumMaxMinPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) {
  return inputs[0];
}

bool CumMaxMinPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs,
    int nbOutputs) {
  switch (pos) {
    // input[0]
    case 0:
      return (inOut[pos].type == nvinfer1::DataType::kFLOAT ||
              inOut[pos].type == nvinfer1::DataType::kINT32) &&
             inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
    // output[0]
    case 1:
      return inOut[pos].type == inOut[0].type &&
             inOut[pos].format == inOut[0].format;
    // output[1]
    case 2:
      return inOut[pos].type == nvinfer1::DataType::kINT32 &&
             inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
    default:
      return false;
  }
}

void CumMaxMinPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs) {}

size_t CumMaxMinPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const {
  int sizeof_dtype = mmcv::getElementSize(outputs[0].type);
}

int CumMaxMinPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *inputDesc,
    const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
    void *const *outputs, void *workSpace, cudaStream_t stream) {
  const void *input = inputs[0];
  void *output_value = outputs[0];
  int *output_index = (int *)outputs[1];

  const int *dims = &(inputDesc[0].dims.d[0]);
  int nbDims = inputDesc[0].dims.nbDims;

  switch (inputDesc[0].type) {
    case nvinfer1::DataType::kFLOAT:
      CumMaxMinForwardLauncher_float((float *)input, (float *)output_value,
                                     output_index, dims, nbDims, mDim,
                                     int(mCumType), stream);
      break;
    case nvinfer1::DataType::kINT32:
      CumMaxMinForwardLauncher_int32((int *)input, (int *)output_value,
                                     output_index, dims, nbDims, mDim,
                                     int(mCumType), stream);
      break;
    default:
      break;
  }

  return 0;
}

nvinfer1::DataType CumMaxMinPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType *inputTypes, int nbInputs) const {
  switch (index) {
    case 0:
      return inputTypes[0];
    case 1:
      return nvinfer1::DataType::kINT32;
    default:
      break;
  }
}

// IPluginV2 Methods
const char *CumMaxMinPluginDynamic::getPluginType() const {
  switch (mCumType) {
    case TRT_CUMCMPTYPE::TRT_CUMMAX:
      return CUMMAX_PLUGIN_NAME;
    case TRT_CUMCMPTYPE::TRT_CUMMIN:
      return CUMMIN_PLUGIN_NAME;
    default:
      return "UnknownCumType";
  }
}

const char *CumMaxMinPluginDynamic::getPluginVersion() const {
  return PLUGIN_VERSION;
}

int CumMaxMinPluginDynamic::getNbOutputs() const { return 2; }

int CumMaxMinPluginDynamic::initialize() { return 0; }

void CumMaxMinPluginDynamic::terminate() {}

size_t CumMaxMinPluginDynamic::getSerializationSize() const {
  return sizeof(mDim) + sizeof(mCumType);
}

void CumMaxMinPluginDynamic::serialize(void *buffer) const {
  serialize_value(&buffer, mDim);
  serialize_value(&buffer, mCumType);
}

void CumMaxMinPluginDynamic::destroy() {
  // This gets called when the network containing plugin is destroyed
  delete this;
}

void CumMaxMinPluginDynamic::setPluginNamespace(const char *libNamespace) {
  mNamespace = libNamespace;
}

const char *CumMaxMinPluginDynamic::getPluginNamespace() const {
  return mNamespace.c_str();
}

CumMaxMinPluginDynamicCreator::CumMaxMinPluginDynamicCreator(
    TRT_CUMCMPTYPE cumType)
    : mCumType(cumType) {
  mPluginAttributes.clear();
  mPluginAttributes.emplace_back(nvinfer1::PluginField("dim"));
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *CumMaxMinPluginDynamicCreator::getPluginName() const {
  return CUMMAXMIN_PLUGIN_NAME;
}

const char *CumMaxMinPluginDynamicCreator::getPluginVersion() const {
  return PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection *
CumMaxMinPluginDynamicCreator::getFieldNames() {
  return &mFC;
}

nvinfer1::IPluginV2 *CumMaxMinPluginDynamicCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) {
  int dim = 0;

  for (int i = 0; i < fc->nbFields; i++) {
    if (fc->fields[i].data == nullptr) {
      continue;
    }
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("dim") == 0) {
      dim = static_cast<const int *>(fc->fields[i].data)[0];
    }
  }

  CumMaxMinPluginDynamic *plugin =
      new CumMaxMinPluginDynamic(name, dim, mCumType);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::IPluginV2 *CumMaxMinPluginDynamicCreator::deserializePlugin(
    const char *name, const void *serialData, size_t serialLength) {
  // This object will be deleted when the network is destroyed, which will
  // call FCPluginDynamic::destroy()
  auto plugin = new CumMaxMinPluginDynamic(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

void CumMaxMinPluginDynamicCreator::setPluginNamespace(
    const char *libNamespace) {
  mNamespace = libNamespace;
}

const char *CumMaxMinPluginDynamicCreator::getPluginNamespace() const {
  return mNamespace.c_str();
}

CumMaxPluginDynamicCreator::CumMaxPluginDynamicCreator()
    : CumMaxMinPluginDynamicCreator(TRT_CUMCMPTYPE::TRT_CUMMAX) {}

const char *CumMaxPluginDynamicCreator::getPluginName() const {
  return CUMMAX_PLUGIN_NAME;
}

CumMinPluginDynamicCreator::CumMinPluginDynamicCreator()
    : CumMaxMinPluginDynamicCreator(TRT_CUMCMPTYPE::TRT_CUMMIN) {}

const char *CumMinPluginDynamicCreator::getPluginName() const {
  return CUMMIN_PLUGIN_NAME;
}
