#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <cstdio>
#include "mnist.hpp"

#define CHECK_CUDNN(expression) \
{ \
    cudnnStatus_t status = (expression); \
    if (status != CUDNN_STATUS_SUCCESS) \
    { \
        std::cerr << "Error on line " << __LINE__ << ": " \
            << cudnnGetErrorString(status) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
}

int main()
{
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));
    printf("Initialized cuDNN\n");

    load_mnist();
    printf("Loaded MNIST\n");
    print_image(train_image[2]);

    constexpr int input_dim = 784;
    constexpr int hidden_dim = 256;
    constexpr int output_dim = 10;
    constexpr int batch_size = 32;

    // Setup input, output, and activation descriptors.
    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t hidden_desc;
    cudnnTensorDescriptor_t output_desc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&hidden_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, 1, 1, input_dim
    ));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        hidden_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, 1, 1, hidden_size
    ));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, 1, 1, output_size
    ));
    
    CHECK_CUDNN(cudnnDestroy(cudnn));
}
