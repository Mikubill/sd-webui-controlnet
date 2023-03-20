// Copyright (c) OpenMMLab. All rights reserved
#include "trt_grid_sampler.hpp"

#include <assert.h>
#include <stdio.h>

#include <chrono>

#include "trt_serialize.hpp"

using mmcv::GridSamplerInterpolation;
using mmcv::GridSamplerPadding;

void grid_sample_float(float *output, const float *input, const float *grid,
                       int *output_dims, int *input_dims, int *grid_dims,
                       int nb_dims, GridSamplerInterpolation interp,
                       GridSamplerPadding padding, bool align_corners,
                       cudaStream_t stream);

namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"grid_sampler"};
}  // namespace

nvinfer1::PluginFieldCollection GridSamplerDynamicCreator::mFC{};
std::vector<nvinfer1::PluginField> GridSamplerDynamicCreator::mPluginAttributes;

GridSamplerDynamic::GridSamplerDynamic(const std::string &name, int mode,
                                       int paddingMode, bool alignCorners)
    : mLayerName(name),
      mMode(mode),
      mPaddingMode(paddingMode),
      mAlignCorners(alignCorners) {}

GridSamplerDynamic::GridSamplerDynamic(const std::string name, const void *data,
                                       size_t length)
    : mLayerName(name) {
  deserialize_value(&data, &length, &mMode);
  deserialize_value(&data, &length, &mPaddingMode);
  deserialize_value(&data, &length, &mAlignCorners);
}

nvinfer1::IPluginV2DynamicExt *GridSamplerDynamic::clone() const {
  GridSamplerDynamic *plugin =
      new GridSamplerDynamic(mLayerName, mMode, mPaddingMode, mAlignCorners);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs GridSamplerDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) {
  nvinfer1::DimsExprs ret;
  ret.nbDims = inputs[0].nbDims;
  ret.d[0] = inputs[0].d[0];
  ret.d[1] = inputs[0].d[1];
  for (int i = 2; i < ret.nbDims; ++i) {
    ret.d[i] = inputs[1].d[i - 1];
  }
  return ret;
}

bool GridSamplerDynamic::supportsFormatCombination(
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

void GridSamplerDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs) {
  // Validate input arguments
}

size_t GridSamplerDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const {
  return 0;
}

int GridSamplerDynamic::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                                const nvinfer1::PluginTensorDesc *outputDesc,
                                const void *const *inputs, void *const *outputs,
                                void *workSpace, cudaStream_t stream) {
  nvinfer1::Dims input_dims = inputDesc[0].dims;
  nvinfer1::Dims grid_dims = inputDesc[1].dims;
  nvinfer1::Dims output_dims = outputDesc[0].dims;

  using mmcv::GridSamplerInterpolation;
  using mmcv::GridSamplerPadding;

  GridSamplerInterpolation interp_mode = GridSamplerInterpolation::Bilinear;
  switch (mMode) {
    case 0:
      interp_mode = GridSamplerInterpolation::Bilinear;
      break;
    case 1:
      interp_mode = GridSamplerInterpolation::Nearest;
      break;
    default:
      break;
  }

  GridSamplerPadding padding_mode = GridSamplerPadding::Zeros;
  switch (mPaddingMode) {
    case 0:
      padding_mode = GridSamplerPadding::Zeros;
      break;

    case 1:
      padding_mode = GridSamplerPadding::Border;
      break;

    case 2:
      padding_mode = GridSamplerPadding::Reflection;
      break;
    default:
      break;
  }

  auto data_type = inputDesc[0].type;

  switch (data_type) {
    case nvinfer1::DataType::kFLOAT:
      grid_sample_float(
          (float *)outputs[0], (float *)inputs[0], (float *)inputs[1],
          &(output_dims.d[0]), &(input_dims.d[0]), &(grid_dims.d[0]),
          input_dims.nbDims, interp_mode, padding_mode, mAlignCorners, stream);
      break;
    default:
      return 1;
      break;
  }

  return 0;
}

nvinfer1::DataType GridSamplerDynamic::getOutputDataType(
    int index, const nvinfer1::DataType *inputTypes, int nbInputs) const {
  return inputTypes[0];
}

// IPluginV2 Methods
const char *GridSamplerDynamic::getPluginType() const { return PLUGIN_NAME; }

const char *GridSamplerDynamic::getPluginVersion() const {
  return PLUGIN_VERSION;
}

int GridSamplerDynamic::getNbOutputs() const { return 1; }

int GridSamplerDynamic::initialize() { return 0; }

void GridSamplerDynamic::terminate() {}

size_t GridSamplerDynamic::getSerializationSize() const {
  return sizeof(mMode) + sizeof(mPaddingMode) + sizeof(mAlignCorners);
}

void GridSamplerDynamic::serialize(void *buffer) const {
  serialize_value(&buffer, mMode);
  serialize_value(&buffer, mPaddingMode);
  serialize_value(&buffer, mAlignCorners);
}

void GridSamplerDynamic::destroy() {
  // This gets called when the network containing plugin is destroyed
  delete this;
}

void GridSamplerDynamic::setPluginNamespace(const char *libNamespace) {
  mNamespace = libNamespace;
}

const char *GridSamplerDynamic::getPluginNamespace() const {
  return mNamespace.c_str();
}

////////////////////// creator /////////////////////////////

GridSamplerDynamicCreator::GridSamplerDynamicCreator() {
  mPluginAttributes.clear();
  mPluginAttributes.emplace_back(nvinfer1::PluginField("interpolation_mode"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("padding_mode"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("align_corners"));
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *GridSamplerDynamicCreator::getPluginName() const {
  return PLUGIN_NAME;
}

const char *GridSamplerDynamicCreator::getPluginVersion() const {
  return PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection *
GridSamplerDynamicCreator::getFieldNames() {
  return &mFC;
}

nvinfer1::IPluginV2 *GridSamplerDynamicCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) {
  int mode = 0;
  int paddingMode = 0;
  bool alignCorners = false;

  for (int i = 0; i < fc->nbFields; i++) {
    if (fc->fields[i].data == nullptr) {
      continue;
    }
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("interpolation_mode") == 0) {
      mode = static_cast<const int *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("padding_mode") == 0) {
      paddingMode = static_cast<const int *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("align_corners") == 0) {
      alignCorners = (bool)(static_cast<const int *>(fc->fields[i].data)[0]);
    }
  }

  GridSamplerDynamic *plugin =
      new GridSamplerDynamic(name, mode, paddingMode, alignCorners);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::IPluginV2 *GridSamplerDynamicCreator::deserializePlugin(
    const char *name, const void *serialData, size_t serialLength) {
  // This object will be deleted when the network is destroyed, which will
  // call FCPluginDynamic::destroy()
  auto plugin = new GridSamplerDynamic(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

void GridSamplerDynamicCreator::setPluginNamespace(const char *libNamespace) {
  mNamespace = libNamespace;
}

const char *GridSamplerDynamicCreator::getPluginNamespace() const {
  return mNamespace.c_str();
}
