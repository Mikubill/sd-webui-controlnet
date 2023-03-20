#ifndef PYTORCH_CPP_HELPER
#define PYTORCH_CPP_HELPER
#include <torch/types.h>

#include <vector>

using namespace at;

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_MLU(x) \
  TORCH_CHECK(x.device().type() == at::kMLU, #x " must be a MLU tensor")
#define CHECK_CPU(x) \
  TORCH_CHECK(x.device().type() == at::kCPU, #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CUDA_INPUT(x) \
  CHECK_CUDA(x);            \
  CHECK_CONTIGUOUS(x)
#define CHECK_MLU_INPUT(x) \
  CHECK_MLU(x);            \
  CHECK_CONTIGUOUS(x)
#define CHECK_CPU_INPUT(x) \
  CHECK_CPU(x);            \
  CHECK_CONTIGUOUS(x)

#endif  // PYTORCH_CPP_HELPER
