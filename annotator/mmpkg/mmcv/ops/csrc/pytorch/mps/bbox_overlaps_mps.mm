// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#include "pytorch_device_registry.hpp"

#include "MPSLibrary.h"
#include "MPSStream.h"
#include "MPSUtils.h"

using at::Tensor;

const static std::string kSourceCode = R"(
#include <metal_math>
#include <metal_stdlib>
using namespace metal;

kernel void bbox_overlap_mps_kernel(constant const float4* bboxes1,
                       constant const float4* bboxes2,
                       device float* ious,
                       constant int& num_bbox1,
                       constant int& num_bbox2,
                       constant int& mode,
                       constant bool& aligned,
                       constant int& offset,
                       uint index [[thread_position_in_grid]])
{
    int base1 = index;
    int base2 = index;
    if(!aligned){
      base1 = index / num_bbox2;
      base2 = index % num_bbox2;
    }

    const float f_offset = float(offset);

    const float4 b1 = bboxes1[base1];
    const float b1_area = (b1[2]-b1[0]+f_offset)*(b1[3]-b1[1]+f_offset);

    const float4 b2 = bboxes2[base2];
    const float b2_area = (b2[2]-b2[0]+f_offset)*(b2[3]-b2[1]+f_offset);

    const float2 left_top = fmax(b1.xy, b2.xy);
    const float2 right_bottom = fmin(b1.zw, b2.zw);
    const float2 wh = fmax(right_bottom - left_top + f_offset, 0.0f);
    const float interS = wh.x * wh.y;

    const float baseS =
        fmax(mode == 0 ? b1_area + b2_area - interS : b1_area, f_offset);
    ious[index] = interS / baseS;
}
)";

void BBoxOverlapsMPSKernelLauncher(const Tensor bboxes1, const Tensor bboxes2, Tensor ious,
                                   const int mode, const bool aligned, const int offset) {
  // get stream
  auto stream = at::mps::getCurrentMPSStream();
  auto library_manager = MPSLibraryManager::getInstance();
  MPSLibrary* library;
  const static std::string kLibraryName = "bbox_overlap";
  if (library_manager->hasLibrary(kLibraryName))
    library = library_manager->getLibrary(kLibraryName);
  else
    library = library_manager->createLibraryFromSouce(kLibraryName, kSourceCode);
  auto func_pso = library->getComputePipelineState("bbox_overlap_mps_kernel");

  // create command buffer and encoder
  MTLCommandBuffer_t command_buffer = stream->commandBuffer();
  MTLComputeCommandEncoder_t compute_encoder = [command_buffer computeCommandEncoder];

  // set pso and buffer
  int output_size = ious.numel();
  int num_bbox1 = bboxes1.size(0);
  int num_bbox2 = bboxes2.size(0);
  int num_elements = output_size;
  setMTLArgs(compute_encoder, func_pso, bboxes1, bboxes2, ious, num_bbox1, num_bbox2, mode, aligned,
             offset);

  // set grid size
  MTLSize grid_size = MTLSizeMake(num_elements, 1, 1);
  NSUInteger thread_group_size_x = func_pso.maxTotalThreadsPerThreadgroup;
  if (thread_group_size_x > num_elements) {
    thread_group_size_x = num_elements;
  }
  MTLSize thread_group_size = MTLSizeMake(thread_group_size_x, 1, 1);

  // encoding
  [compute_encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
  [compute_encoder endEncoding];

  // commit, not sure if flush is required
  stream->commit(false);
}

void bbox_overlaps_mps(const Tensor bboxes1, const Tensor bboxes2, Tensor ious, const int mode,
                       const bool aligned, const int offset) {
  BBoxOverlapsMPSKernelLauncher(bboxes1, bboxes2, ious, mode, aligned, offset);
}

void bbox_overlaps_impl(const Tensor bboxes1, const Tensor bboxes2, Tensor ious, const int mode,
                        const bool aligned, const int offset);
REGISTER_DEVICE_IMPL(bbox_overlaps_impl, MPS, bbox_overlaps_mps);
