// Copyright (c) OpenMMLab. All rights reserved
#include "trt_roi_align.hpp"

#include <assert.h>

#include <chrono>

#include "trt_serialize.hpp"

extern void TRTRoIAlignForwardCUDAKernelLauncher_float(
    const float *input, const float *rois, float *output, float *argmax_y,
    float *argmax_x, int output_size, int channels, int height, int width,
    int aligned_height, int aligned_width, float spatial_scale,
    int sampling_ratio, int pool_mode, bool aligned, cudaStream_t stream);

namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"MMCVRoiAlign"};
}  // namespace

nvinfer1::PluginFieldCollection RoIAlignPluginDynamicCreator::mFC{};
std::vector<nvinfer1::PluginField>
    RoIAlignPluginDynamicCreator::mPluginAttributes;

RoIAlignPluginDynamic::RoIAlignPluginDynamic(const std::string &name,
                                             int outWidth, int outHeight,
                                             float spatialScale,
                                             int sampleRatio, int poolMode,
                                             bool aligned)
    : mLayerName(name),
      mOutWidth(outWidth),
      mOutHeight(outHeight),
      mSpatialScale(spatialScale),
      mSampleRatio(sampleRatio),
      mPoolMode(poolMode),
      mAligned(aligned) {}

RoIAlignPluginDynamic::RoIAlignPluginDynamic(const std::string name,
                                             const void *data, size_t length)
    : mLayerName(name) {
  deserialize_value(&data, &length, &mOutWidth);
  deserialize_value(&data, &length, &mOutHeight);
  deserialize_value(&data, &length, &mSpatialScale);
  deserialize_value(&data, &length, &mSampleRatio);
  deserialize_value(&data, &length, &mPoolMode);
  deserialize_value(&data, &length, &mAligned);
}

nvinfer1::IPluginV2DynamicExt *RoIAlignPluginDynamic::clone() const {
  RoIAlignPluginDynamic *plugin = new RoIAlignPluginDynamic(
      mLayerName, mOutWidth, mOutHeight, mSpatialScale, mSampleRatio, mPoolMode,
      mAligned);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs RoIAlignPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) {
  nvinfer1::DimsExprs ret;
  ret.nbDims = 4;
  ret.d[0] = inputs[1].d[0];
  ret.d[1] = inputs[0].d[1];
  ret.d[2] = exprBuilder.constant(mOutHeight);
  ret.d[3] = exprBuilder.constant(mOutWidth);

  return ret;
}

bool RoIAlignPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs,
    int nbOutputs) {
  return inOut[pos].type == nvinfer1::DataType::kFLOAT &&
         inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
}

void RoIAlignPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs) {}

size_t RoIAlignPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const {
  size_t output_size = 0;
  size_t word_size = 0;
  switch (mPoolMode) {
    case 0:  // max
      output_size = outputs[0].dims.d[0] * outputs[0].dims.d[1] *
                    outputs[0].dims.d[2] * outputs[0].dims.d[3];
      word_size = mmcv::getElementSize(outputs[0].type);
      return output_size * word_size * 2;
      break;
    case 1:
      return 0;
      break;
    default:
      return 0;
  }
  return 0;
}

int RoIAlignPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                                   const nvinfer1::PluginTensorDesc *outputDesc,
                                   const void *const *inputs,
                                   void *const *outputs, void *workSpace,
                                   cudaStream_t stream) {
  int channels = inputDesc[0].dims.d[1];
  int height = inputDesc[0].dims.d[2];
  int width = inputDesc[0].dims.d[3];

  int output_size = outputDesc[0].dims.d[0] * outputDesc[0].dims.d[1] *
                    outputDesc[0].dims.d[2] * outputDesc[0].dims.d[3];
  int word_size = mmcv::getElementSize(outputDesc[0].type);

  const void *feat = inputs[0];
  const void *rois = inputs[1];
  void *output = outputs[0];
  void *argmax_y = nullptr;
  void *argmax_x = nullptr;

  switch (mPoolMode) {
    case 0:  // max
      argmax_y = workSpace;
      argmax_x = argmax_y + output_size * word_size;
      break;
    case 1:  // avg
      break;
  }

  switch (outputDesc[0].type) {
    case nvinfer1::DataType::kFLOAT:
      TRTRoIAlignForwardCUDAKernelLauncher_float(
          (const float *)feat, (const float *)rois, (float *)output,
          (float *)argmax_y, (float *)argmax_x, output_size, channels, height,
          width, mOutHeight, mOutWidth, mSpatialScale, mSampleRatio, mPoolMode,
          mAligned, stream);
      break;

    default:
      break;
  }

  return 0;
}

nvinfer1::DataType RoIAlignPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType *inputTypes, int nbInputs) const {
  return inputTypes[0];
}

// IPluginV2 Methods
const char *RoIAlignPluginDynamic::getPluginType() const { return PLUGIN_NAME; }

const char *RoIAlignPluginDynamic::getPluginVersion() const {
  return PLUGIN_VERSION;
}

int RoIAlignPluginDynamic::getNbOutputs() const { return 1; }

int RoIAlignPluginDynamic::initialize() { return 0; }

void RoIAlignPluginDynamic::terminate() {}

size_t RoIAlignPluginDynamic::getSerializationSize() const {
  return sizeof(mOutWidth) + sizeof(mOutHeight) + sizeof(mSpatialScale) +
         sizeof(mSampleRatio) + sizeof(mPoolMode) + sizeof(mAligned);
}

void RoIAlignPluginDynamic::serialize(void *buffer) const {
  serialize_value(&buffer, mOutWidth);
  serialize_value(&buffer, mOutHeight);
  serialize_value(&buffer, mSpatialScale);
  serialize_value(&buffer, mSampleRatio);
  serialize_value(&buffer, mPoolMode);
  serialize_value(&buffer, mAligned);
}

void RoIAlignPluginDynamic::destroy() {
  // This gets called when the network containing plugin is destroyed
  delete this;
}

void RoIAlignPluginDynamic::setPluginNamespace(const char *libNamespace) {
  mNamespace = libNamespace;
}

const char *RoIAlignPluginDynamic::getPluginNamespace() const {
  return mNamespace.c_str();
}

////////////////////// creator /////////////////////////////

RoIAlignPluginDynamicCreator::RoIAlignPluginDynamicCreator() {
  mPluginAttributes.emplace_back(nvinfer1::PluginField("output_height"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("output_width"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("spatial_scale"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("sampling_ratio"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("mode"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("aligned"));
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *RoIAlignPluginDynamicCreator::getPluginName() const {
  return PLUGIN_NAME;
}

const char *RoIAlignPluginDynamicCreator::getPluginVersion() const {
  return PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection *
RoIAlignPluginDynamicCreator::getFieldNames() {
  return &mFC;
}

nvinfer1::IPluginV2 *RoIAlignPluginDynamicCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) {
  int outWidth = 7;
  int outHeight = 7;
  float spatialScale = 1.0;
  int sampleRatio = 0;
  int poolMode = -1;
  bool aligned = true;
  for (int i = 0; i < fc->nbFields; i++) {
    if (fc->fields[i].data == nullptr) {
      continue;
    }
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("output_height") == 0) {
      outHeight = static_cast<const int *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("output_width") == 0) {
      outWidth = static_cast<const int *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("spatial_scale") == 0) {
      spatialScale = static_cast<const float *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("sampling_ratio") == 0) {
      sampleRatio = static_cast<const int *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("mode") == 0) {
      int data_size = fc->fields[i].length;
      const char *data_start = static_cast<const char *>(fc->fields[i].data);
      std::string poolModeStr(data_start, data_size);
      if (poolModeStr == "avg") {
        poolMode = 1;
      } else if (poolModeStr == "max") {
        poolMode = 0;
      } else {
        std::cout << "Unknown pool mode \"" << poolModeStr << "\"."
                  << std::endl;
      }
      assert(poolMode >= 0);
    }

    if (field_name.compare("aligned") == 0) {
      int aligned_int = static_cast<const int *>(fc->fields[i].data)[0];
      aligned = aligned_int != 0;
    }
  }

  assert(outHeight > 0);
  assert(outWidth > 0);
  assert(spatialScale > 0.);
  assert(poolMode >= 0);

  RoIAlignPluginDynamic *plugin = new RoIAlignPluginDynamic(
      name, outWidth, outHeight, spatialScale, sampleRatio, poolMode, aligned);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::IPluginV2 *RoIAlignPluginDynamicCreator::deserializePlugin(
    const char *name, const void *serialData, size_t serialLength) {
  auto plugin = new RoIAlignPluginDynamic(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

void RoIAlignPluginDynamicCreator::setPluginNamespace(
    const char *libNamespace) {
  mNamespace = libNamespace;
}

const char *RoIAlignPluginDynamicCreator::getPluginNamespace() const {
  return mNamespace.c_str();
}
