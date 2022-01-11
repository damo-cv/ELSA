//
// This file is modified from DDFNet: https://github.com/thefoxofsky/ddfnet
//
#include <ATen/ATen.h>
#include <torch/extension.h>

#include <cmath>
#include <vector>

#ifdef WITH_CUDA
    int elsa_faster_forward_cuda(
        at::Tensor features, at::Tensor channel_mul, at::Tensor channel_add, at::Tensor spatial_filter,
        int kernel_size, int dilation, int stride, at::Tensor output);
#endif

int elsa_faster_forward(
    at::Tensor features,at::Tensor channel_mul, at::Tensor channel_add, at::Tensor spatial_filter,
    int kernel_size, int dilation, int stride, at::Tensor output){
    if (features.device().is_cuda()){
        #ifdef WITH_CUDA
            return elsa_faster_forward_cuda(
                features, channel_mul, channel_add, spatial_filter,
                kernel_size, dilation, stride, output);
        #else
            AT_ERROR("elsa operation is not compiled with GPU support");
        #endif
    }
    AT_ERROR("elsa operation is not implemented on CPU");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &elsa_faster_forward, "elsa faster forward");
}
