#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <cstdio>
#include "mnist.hpp"

#define CHECK_CUDNN(expression) \
{ \
    cudnnStatus_t status = (expression); \
    if(status != CUDNN_STATUS_SUCCESS) \
    { \
        std::cerr << "Error on line " << __LINE__ << ": " \
            << cudnnGetErrorString(status) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUDA(expression) \
{ \
    cudaError_t error = (expression); \
    if(error != 0) \
    { \
        std::cerr << "Error on line " << __LINE__ << ": " \
            << cudaGetErrorString(error) << std::endl; \
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

    // Setup input, output, and hidden descriptors.
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
        hidden_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, 1, 1, hidden_dim
    ));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, 1, 1, output_dim
    ));
 
    // Create fully connected layer descriptors.
    cudnnTensorDescriptor_t fc1_filter_desc;
    cudnnTensorDescriptor_t fc2_filter_desc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&fc1_filter_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&fc2_filter_desc));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        fc1_filter_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, hidden_dim, 1, input_dim, 1 
    ));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        fc2_filter_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output_dim, 1, hidden_dim, 1 
    ));

    // Setup ReLU.
    cudnnActivationDescriptor_t relu_desc;
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&relu_desc));
    CHECK_CUDNN(cudnnSetActivationDescriptor(
        relu_desc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0
    ));

    // Setup softmax descriptor.
    // CUDNN_SOFTMAX_ACCURATE represents "safe softmax",
    // (max value is subtracted from exponents to prevent overflow).
    cudnnSoftmaxAlgorithm_t softmax_algo = CUDNN_SOFTMAX_ACCURATE;
    cudnnSoftmaxMode_t softmax_mode = CUDNN_SOFTMAX_MODE_INSTANCE;
    
    // Allocate memory for weights and biases.
    float* d_fc1_weights;
    float* d_fc2_weights;
    float* d_fc1_bias;
    float* d_fc2_bias;
    float* d_input;
    float* d_hidden;
    float* d_output;

    CHECK_CUDA(cudaMalloc(&d_fc1_weights, sizeof(float) * input_dim * hidden_dim));
    CHECK_CUDA(cudaMalloc(&d_fc2_weights, sizeof(float) * hidden_dim * output_dim));
    CHECK_CUDA(cudaMalloc(&d_fc1_bias, sizeof(float) * hidden_dim));
    CHECK_CUDA(cudaMalloc(&d_fc2_bias, sizeof(float) * output_dim));
    CHECK_CUDA(cudaMalloc(&d_input, sizeof(float) * batch_size * input_dim));
    CHECK_CUDA(cudaMalloc(&d_hidden, sizeof(float) * batch_size * hidden_dim));
    CHECK_CUDA(cudaMalloc(&d_output, sizeof(float) * batch_size * output_dim));

    // Define forward pass. 
    float alpha = 1.0f, beta = 0.0f;

    // First fully connected layer + ReLU.
    CHECK_CUDNN(cudnnFullyConnectedForward(
        cudnn,
        &alpha,
        input_desc, d_input,
        fc1_filter_desc, d_fc1_weights,
        d_fc1_bias,
        &beta,
        hidden_desc, d_hidden
    ));

    CHECK_CUDNN(cudnnDestroy(cudnn));
}
