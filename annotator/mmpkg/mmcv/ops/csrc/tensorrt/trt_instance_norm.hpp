// Modified from:
// https://github.com/NVIDIA/TensorRT/blob/master/plugin/instanceNormalizationPlugin/instanceNormalizationPlugin.h

#ifndef TRT_INSTANCE_NORMALIZATION_PLUGIN_H
#define TRT_INSTANCE_NORMALIZATION_PLUGIN_H
#include <cudnn.h>

#include <iostream>
#include <string>
#include <vector>

#include "trt_plugin_helper.hpp"

typedef unsigned short half_type;

class InstanceNormalizationDynamic final
    : public nvinfer1::IPluginV2DynamicExt {
 public:
  InstanceNormalizationDynamic(const std::string& name, float epsilon);

  InstanceNormalizationDynamic(const std::string& name, void const* serialData,
                               size_t serialLength);

  InstanceNormalizationDynamic() = delete;

  ~InstanceNormalizationDynamic() override;

  int getNbOutputs() const override;

  // DynamicExt plugins returns DimsExprs class instead of Dims
  nvinfer1::DimsExprs getOutputDimensions(
      int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
      nvinfer1::IExprBuilder& exprBuilder) override;

  int initialize() override;

  void terminate() override;

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nbOutputs) const override;

  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs, void* const* outputs, void* workspace,
              cudaStream_t stream) override;

  size_t getSerializationSize() const override;

  void serialize(void* buffer) const override;

  // DynamicExt plugin supportsFormat update.
  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* inOut,
                                 int nbInputs, int nbOutputs) override;

  const char* getPluginType() const override;

  const char* getPluginVersion() const override;

  void destroy() override;

  nvinfer1::IPluginV2DynamicExt* clone() const override;

  void setPluginNamespace(const char* pluginNamespace) override;

  const char* getPluginNamespace() const override;

  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* inputTypes,
                                       int nbInputs) const override;

  void attachToContext(cudnnContext* cudnn, cublasContext* cublas,
                       nvinfer1::IGpuAllocator* allocator) override;

  void detachFromContext() override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int nbOutputs) override;

 private:
  const std::string mLayerName;
  float mEpsilon{};
  cudnnHandle_t _cudnn_handle{};
  cudnnTensorDescriptor_t _x_desc{}, _y_desc{}, _b_desc{};
  std::string mPluginNamespace{};
};

class InstanceNormalizationDynamicCreator : public nvinfer1::IPluginCreator {
 public:
  InstanceNormalizationDynamicCreator();

  ~InstanceNormalizationDynamicCreator() override = default;

  const char* getPluginName() const override;

  const char* getPluginVersion() const override;

  const nvinfer1::PluginFieldCollection* getFieldNames() override;

  nvinfer1::IPluginV2DynamicExt* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* fc) override;

  nvinfer1::IPluginV2DynamicExt* deserializePlugin(
      const char* name, const void* serialData, size_t serialLength) override;

  void setPluginNamespace(const char* pluginNamespace) override;

  const char* getPluginNamespace() const override;

 private:
  static nvinfer1::PluginFieldCollection mFC;
  static std::vector<nvinfer1::PluginField> mPluginAttributes;
  std::string mNamespace;
};

#endif  // TRT_INSTANCE_NORMALIZATION_PLUGIN_H
