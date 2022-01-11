//
// This file is modified from DDFNet: https://github.com/thefoxofsky/ddfnet
//
#include <ATen/ATen.h>
#include <torch/extension.h>

#include <cmath>
#include <vector>

#ifdef WITH_CUDA
    int elsa_forward_cuda(
        at::Tensor features, at::Tensor channel_mul, at::Tensor channel_add, at::Tensor spatial_filter,
        int kernel_size, int dilation, int stride, at::Tensor output);

    int elsa_backward_cuda(
        at::Tensor top_grad, at::Tensor features,
        at::Tensor channel_mul, at::Tensor channel_add, at::Tensor spatial_filter,
        int kernel_size, int dilation, int stride,
        at::Tensor rtop_grad, at::Tensor rbottom_grad,
        at::Tensor rspatial_filter_grad, at::Tensor bottom_grad,
        at::Tensor channel_mul_grad, at::Tensor channel_add_grad, at::Tensor spatial_filter_grad);
#endif

int elsa_forward(
    at::Tensor features,at::Tensor channel_mul, at::Tensor channel_add, at::Tensor spatial_filter,
    int kernel_size, int dilation, int stride, at::Tensor output){
    if (features.device().is_cuda()){
        #ifdef WITH_CUDA
            return elsa_forward_cuda(
                features, channel_mul, channel_add, spatial_filter,
                kernel_size, dilation, stride, output);
        #else
            AT_ERROR("elsa operation is not compiled with GPU support");
        #endif
    }
    AT_ERROR("elsa operation is not implemented on CPU");
}

int elsa_backward(
    at::Tensor top_grad, at::Tensor features,
    at::Tensor channel_mul, at::Tensor channel_add, at::Tensor spatial_filter,
    int kernel_size, int dilation, int stride,
    at::Tensor rtop_grad, at::Tensor rbottom_grad,
    at::Tensor rspatial_filter_grad, at::Tensor bottom_grad,
    at::Tensor channel_mul_grad, at::Tensor channel_add_grad, at::Tensor spatial_filter_grad){
    if (top_grad.device().is_cuda()){
        #ifdef WITH_CUDA
            return elsa_backward_cuda(
                top_grad, features, channel_mul, channel_add, spatial_filter,
                kernel_size, dilation, stride,
                rtop_grad, rbottom_grad, rspatial_filter_grad,
                bottom_grad, channel_mul_grad, channel_add_grad, spatial_filter_grad);
        #else
            AT_ERROR("elsa operation is not compiled with GPU support");
        #endif
    }
    AT_ERROR("elsa operation is not implemented on CPU");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &elsa_forward, "elsa forward");
  m.def("backward", &elsa_backward, "elsa backward");
}
