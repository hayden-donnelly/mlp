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
    print_image(train_image[2]);

    constexpr int input_dim = 784;
    constexpr int hidden_dim = 256;
    constexpr int output_dim = 10;
}
