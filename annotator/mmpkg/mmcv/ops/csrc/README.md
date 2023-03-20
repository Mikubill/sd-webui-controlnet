# Code Structure of CUDA operators

This folder contains all non-python code for MMCV custom ops. Please follow the same architecture if you want to add new ops.

## Directories Tree

```folder
.
├── common
│   ├── box_iou_rotated_utils.hpp
│   ├── parrots_cpp_helper.hpp
│   ├── parrots_cuda_helper.hpp
│   ├── pytorch_cpp_helper.hpp
│   ├── pytorch_cuda_helper.hpp
│   ├── pytorch_device_registry.hpp
│   ├── cuda
│   │   ├── common_cuda_helper.hpp
│   │   ├── parrots_cudawarpfunction.cuh
│   │   ├── ...
│   │   └── ops_cuda_kernel.cuh
|   ├── mps
│   │   ├── MPSLibrary.h
│   │   ├── ...
│   │   └── MPSUtils.h
|   ├── mlu
│   │   └── ...
|   └── utils
│   │   └── ...
├── onnxruntime
│   ├── onnxruntime_register.h
│   ├── onnxruntime_session_options_config_keys.h
│   ├── ort_mmcv_utils.h
│   ├── ...
│   ├── onnx_ops.h
│   └── cpu
│       ├── onnxruntime_register.cpp
│       ├── ...
│       └── onnx_ops_impl.cpp
├── parrots
│   ├── ...
│   ├── ops.cpp
│   ├── ops_parrots.cpp
│   └── ops_pytorch.h
├── pytorch
│   ├── info.cpp
│   ├── pybind.cpp
│   ├── ...
│   ├── ops.cpp
│   ├── cuda
│   │   ├── ...
│   │   └── ops_cuda.cu
│   ├── cpu
│   │   ├── ...
│   │   └── ops.cpp
│   ├── mps
│   │   ├── ...
│   |   └── op_mps.mm
│   └── mlu
│       ├── ...
│       └── op_mlu.cpp
└── tensorrt
    ├── trt_cuda_helper.cuh
    ├── trt_plugin_helper.hpp
    ├── trt_plugin.hpp
    ├── trt_serialize.hpp
    ├── ...
    ├── trt_ops.hpp
    └── plugins
        ├── trt_cuda_helper.cu
        ├── trt_plugin.cpp
        ├── ...
        ├── trt_ops.cpp
        └── trt_ops_kernel.cu
```

## Components

- `common`: This directory contains all tools and shared codes.
  - `cuda`: The cuda kernels which can be shared by all backends. **HIP** kernel is also here since they have similar syntax.
  - `mps`: The tools used to support MPS ops. **NOTE** that MPS support is **experimental**.
  - `mlu`: The MLU kernels used to support [Cambricon](https://www.cambricon.com/) device.
  - `utils`: The kernels and utils of spconv.
- `onnxruntime`: **ONNX Runtime** support for custom ops. Has been deprecated, please try the latest custom ops in [MMDeploy](https://github.com/open-mmlab/mmdeploy).
  - `cpu`: CPU implementation of supported ops.
- `parrots`: **Parrots** is a deep learning frame for model training and inference. Parrots custom ops are placed in this directory.
- `pytorch`: **PyTorch** custom ops are supported by binding C++ to Python with **pybind11**. The ops implementation and binding codes are placed in this directory.
  - `cuda`: This directory contains cuda kernel launchers, which feed memory pointers of tensor to the cuda kernel in `common/cuda`. The launchers provide c++ interface of cuda implementation of corresponding custom ops.
  - `cpu`: This directory contain cpu implementations of corresponding custom ops.
  - `mlu`: This directory contain launchers of each MLU kernels.
  - `mps`: MPS ops implementation and launchers.
- `tensorrt`: **TensorRT** support for custom ops. Has been deprecated, please try the latest custom ops in [MMDeploy](https://github.com/open-mmlab/mmdeploy).
  - `plugins`: This directory contains the implementation of the supported custom ops. Some ops might also use shared cuda kernel in `common/cuda`.

## How to add new PyTorch ops?

1. (Optional) Add shared kernel in `common` to support special hardware platform.

   ```c++
   // src/common/cuda/new_ops_cuda_kernel.cuh

   template <typename T>
   __global__ void new_ops_forward_cuda_kernel(const T* input, T* output, ...) {
       // forward here
   }

   ```

   Add cuda kernel launcher in `pytorch/cuda`.

   ```c++
   // src/pytorch/cuda
   #include <new_ops_cuda_kernel.cuh>

   void NewOpsForwardCUDAKernelLauncher(Tensor input, Tensor output, ...){
       // initialize
       at::cuda::CUDAGuard device_guard(input.device());
       cudaStream_t stream = at::cuda::getCurrentCUDAStream();
       ...
       AT_DISPATCH_FLOATING_TYPES_AND_HALF(
           input.scalar_type(), "new_ops_forward_cuda_kernel", ([&] {
               new_ops_forward_cuda_kernel<scalar_t>
                   <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                       input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),...);
           }));
       AT_CUDA_CHECK(cudaGetLastError());
   }
   ```

2. Register implementation for different devices.

   ```c++
   // src/pytorch/cuda/cudabind.cpp
   ...

   Tensor new_ops_forward_cuda(Tensor input, Tensor output, ...){
       // implement cuda forward here
       // use `NewOpsForwardCUDAKernelLauncher` here
   }
   // declare interface here.
   Tensor new_ops_forward_impl(Tensor input, Tensor output, ...);
   // register the implementation for given device (CUDA here).
   REGISTER_DEVICE_IMPL(new_ops_forward_impl, CUDA, new_ops_forward_cuda);
   ```

3. Add ops implementation in `pytorch` directory. Select different implementations according to device type.

   ```c++
   // src/pytorch/new_ops.cpp
   Tensor new_ops_forward_impl(Tensor input, Tensor output, ...){
       // dispatch the implementation according to the device type of input.
       DISPATCH_DEVICE_IMPL(new_ops_forward_impl, input, output, ...);
   }
   ...

   Tensor new_ops_forward(Tensor input, Tensor output, ...){
       return new_ops_forward_impl(input, output, ...);
   }
   ```

4. Binding the implementation in `pytorch/pybind.cpp`

   ```c++
   // src/pytorch/pybind.cpp

   ...

   Tensor new_ops_forward(Tensor input, Tensor output, ...);

   ...

   // bind with pybind11
   m.def("new_ops_forward", &new_ops_forward, "new_ops_forward",
           py::arg("input"), py::arg("output"), ...);

   ...

   ```

5. Build MMCV again. Enjoy new ops in python

   ```python
   from ..utils import ext_loader
   ext_module = ext_loader.load_ext('_ext', ['new_ops_forward'])

   ...

   ext_module.new_ops_forward(input, output, ...)

   ```
