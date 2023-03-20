// Copyright (c) OpenMMLab. All rights reserved
#include "trt_scatternd.hpp"

#include <assert.h>
#include <stdio.h>

#include <chrono>

#include "trt_serialize.hpp"

extern void TRTONNXScatterNDKernelLauncher_float(
    const float *data, const int *indices, const float *update, const int *dims,
    int nbDims, const int *indices_dims, int indice_nbDims, float *output,
    cudaStream_t stream);

extern void TRTONNXScatterNDKernelLauncher_int32(
    const int *data, const int *indices, const int *update, const int *dims,
    int nbDims, const int *indices_dims, int indice_nbDims, int *output,
    cudaStream_t stream);

namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"ScatterND"};
}  // namespace

nvinfer1::PluginFieldCollection ONNXScatterNDDynamicCreator::mFC{};
std::vector<nvinfer1::PluginField>
    ONNXScatterNDDynamicCreator::mPluginAttributes;

ONNXScatterNDDynamic::ONNXScatterNDDynamic(const std::string &name)
    : mLayerName(name) {}

ONNXScatterNDDynamic::ONNXScatterNDDynamic(const std::string name,
                                           const void *data, size_t length)
    : mLayerName(name) {}

nvinfer1::IPluginV2DynamicExt *ONNXScatterNDDynamic::clone() const {
  ONNXScatterNDDynamic *plugin = new ONNXScatterNDDynamic(mLayerName);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs ONNXScatterNDDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) {
  return inputs[0];
}

bool ONNXScatterNDDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs,
    int nbOutputs) {
  if (pos < nbInputs) {
    switch (pos) {
      case 0:
        // data
        return (inOut[pos].type == nvinfer1::DataType::kFLOAT &&
                inOut[pos].format == nvinfer1::TensorFormat::kLINEAR) ||
               (inOut[pos].type == nvinfer1::DataType::kINT32 &&
                inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
      case 1:
        // indices
        return inOut[pos].type == nvinfer1::DataType::kINT32 &&
               inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
      case 2:
        // updates
        return inOut[pos].type == inOut[0].type &&
               inOut[pos].format == inOut[0].format;
      default:
        return true;
    }
  } else {
    switch (pos - nbInputs) {
      case 0:
        // output
        return inOut[pos].type == inOut[0].type &&
               inOut[pos].format == inOut[0].format;
      default:
        return true;
    }
  }
  return true;
}

void ONNXScatterNDDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs) {}

size_t ONNXScatterNDDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const {
  return 0;
}

int ONNXScatterNDDynamic::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                                  const nvinfer1::PluginTensorDesc *outputDesc,
                                  const void *const *inputs,
                                  void *const *outputs, void *workSpace,
                                  cudaStream_t stream) {
  const int *dims = &(inputDesc[0].dims.d[0]);
  const int *indices_dims = &(inputDesc[1].dims.d[0]);
  int nbDims = inputDesc[0].dims.nbDims;
  int indice_nbDims = inputDesc[1].dims.nbDims;

  const void *data = inputs[0];
  const void *indices = inputs[1];
  const void *update = inputs[2];
  void *output = outputs[0];

  auto data_type = inputDesc[0].type;

  switch (data_type) {
    case nvinfer1::DataType::kFLOAT:
      TRTONNXScatterNDKernelLauncher_float(
          (float *)data, (int *)indices, (float *)update, dims, nbDims,
          indices_dims, indice_nbDims, (float *)output, stream);
      break;

    case nvinfer1::DataType::kINT32:
      TRTONNXScatterNDKernelLauncher_int32(
          (int *)data, (int *)indices, (int *)update, dims, nbDims,
          indices_dims, indice_nbDims, (int *)output, stream);
      break;
    default:
      break;
  }

  return 0;
}

nvinfer1::DataType ONNXScatterNDDynamic::getOutputDataType(
    int index, const nvinfer1::DataType *inputTypes, int nbInputs) const {
  return inputTypes[0];
}

// IPluginV2 Methods
const char *ONNXScatterNDDynamic::getPluginType() const { return PLUGIN_NAME; }

const char *ONNXScatterNDDynamic::getPluginVersion() const {
  return PLUGIN_VERSION;
}

int ONNXScatterNDDynamic::getNbOutputs() const { return 1; }

int ONNXScatterNDDynamic::initialize() { return 0; }

void ONNXScatterNDDynamic::terminate() {}

size_t ONNXScatterNDDynamic::getSerializationSize() const { return 0; }

void ONNXScatterNDDynamic::serialize(void *buffer) const {}

void ONNXScatterNDDynamic::destroy() {
  // This gets called when the network containing plugin is destroyed
  delete this;
}

void ONNXScatterNDDynamic::setPluginNamespace(const char *libNamespace) {
  mNamespace = libNamespace;
}

const char *ONNXScatterNDDynamic::getPluginNamespace() const {
  return mNamespace.c_str();
}

////////////////////// creator /////////////////////////////

ONNXScatterNDDynamicCreator::ONNXScatterNDDynamicCreator() {
  mPluginAttributes.clear();
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *ONNXScatterNDDynamicCreator::getPluginName() const {
  return PLUGIN_NAME;
}

const char *ONNXScatterNDDynamicCreator::getPluginVersion() const {
  return PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection *
ONNXScatterNDDynamicCreator::getFieldNames() {
  return &mFC;
}

nvinfer1::IPluginV2 *ONNXScatterNDDynamicCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) {
  ONNXScatterNDDynamic *plugin = new ONNXScatterNDDynamic(name);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::IPluginV2 *ONNXScatterNDDynamicCreator::deserializePlugin(
    const char *name, const void *serialData, size_t serialLength) {
  auto plugin = new ONNXScatterNDDynamic(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

void ONNXScatterNDDynamicCreator::setPluginNamespace(const char *libNamespace) {
  mNamespace = libNamespace;
}

const char *ONNXScatterNDDynamicCreator::getPluginNamespace() const {
  return mNamespace.c_str();
}
