#include <cuda.h>
#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "mnist.hpp"

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

void device_to_host_and_print(int height, int width, float* d_A)
{
    size_t mat_size = sizeof(float) * height * width;
    float* h_A = (float*)malloc(mat_size);
    cudaMemcpy(h_A, d_A, mat_size, cudaMemcpyDeviceToHost);
    for(int i = 0; i < height; ++i)
    {
        printf("[");
        for(int k = 0; k < width; ++k)
        {
            printf("%f ", h_A[i*width + k]);
        }
        printf("]\n");
    }
    free(h_A);
}

struct mlp_t
{
    int input_dim;
    int hidden_dim;
    int output_dim;
    int batch_size;
    
    float* fc1_w;
    float* fc1_b;
    float* fc2_w;
    float* fc2_b;
    
    float* input;
    float* fc1_w_inter;
    float* fc1_b_inter;
    float* relu_inter;
    float* fc2_w_inter;
    float* fc2_b_inter;
    float* output;
};

template<int tile_width>
__global__ void fc_forward_kernel(
    const float* W, // Shape: (input_dim, output_dim)
    const float* X, // Shape: (batch_size, input_dim)
    float* Y,       // Shape: (batch_size, output_dim)
    int input_dim, int output_dim, int batch_size
){
    __shared__ float X_s[tile_width][tile_width];
    __shared__ float W_s[tile_width][tile_width];

    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;

    int row = block_y * tile_width + thread_y;
    int col = block_x * tile_width + thread_x;

    float Y_val = 0.0f;
    for(int ph = 0; ph < ceil(output_dim/(float)tile_width); ++ph)
    {
        // Load W tile into shared memory.
        if(row < output_dim && ph*tile_width + thread_x < input_dim)
        {
            // Tiled vertically.
            W_s[thread_y][thread_x] = W[(ph*tile_width + thread_y)*output_dim + col];
        }
        else
        {
            W_s[thread_y][thread_x] = 0.0f;
        }

        // Load X tile into shared memory.
        if(col < input_dim && ph*tile_width + thread_y < output_dim)
        {
            // Tiled horizontally. 
            X_s[thread_y][thread_x] = X[row*input_dim + ph*tile_width + thread_x];
        }
        else
        {
            X_s[thread_y][thread_x] = 0.0f;
        }
        __syncthreads();
    
        // Inner loop dot product.
        for(int k = 0; k < tile_width; ++k)
        {
            Y_val += X_s[thread_y][k] * W_s[k][thread_x];
        }
        __syncthreads();
    }

    if(row < output_dim && col < input_dim)
    {
        Y[row*output_dim + col] = Y_val;
    }
}

template<int tile_width> 
void fc_forward_launch(
    const float* W, const float* X, float* Y,
    int input_dim, int output_dim, int batch_size 
){
    const int block_size = 32;
    dim3 grid_dim((output_dim + tile_width - 1) / tile_width, (batch_size + tile_width - 1) / tile_width);
    dim3 block_dim(tile_width, tile_width);
    //printf("block_dim: (%d, %d)\ngrid_dim: (%d, %d)\n", block_dim.x, block_dim.y, grid_dim.x, grid_dim.y);
    fc_forward_kernel<tile_width><<<grid_dim, block_dim>>>(W, X, Y, input_dim, output_dim, batch_size);
}

__global__ void relu_forward_kernel(const float* X, float* Y, int input_dim, int batch_size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(col < input_dim && row < batch_size)
    {
        int idx = row*input_dim + col;
        float X_val = X[idx];
        Y[row*input_dim + col] = (X_val > 0.0f) ? X_val : 0.0f;
    }
}

void relu_forward_launch(const float* X, float* Y, int input_dim, int batch_size)
{
    const int block_x = (input_dim + 31) / 32;
    dim3 grid_dim((input_dim + block_x - 1) / block_x, batch_size);
    dim3 block_dim(block_x, 1);
    relu_forward_kernel<<<grid_dim, block_dim>>>(X, Y, input_dim, batch_size);
}

template<int tile_width> 
void forward_pass(mlp_t* mlp)
{
    fc_forward_launch<tile_width>(
        (const float*)mlp->fc1_w, (const float*)mlp->input, mlp->fc1_w_inter,
        mlp->input_dim, mlp->hidden_dim, mlp->batch_size
    );
    relu_forward_launch((const float*)mlp->fc1_w_inter, mlp->relu_inter, mlp->hidden_dim, mlp->batch_size);
    printf("fc1_w_inter:\n");
    device_to_host_and_print(mlp->batch_size, mlp->hidden_dim, mlp->fc1_w_inter);
    printf("\n");
    printf("relu_inter:\n");
    device_to_host_and_print(mlp->batch_size, mlp->hidden_dim, mlp->relu_inter);
    printf("\n");
    // TODO:
    // bias
    // relu
    fc_forward_launch<tile_width>(
        (const float*)mlp->fc2_w, (const float*)mlp->fc1_w_inter, mlp->output,
        mlp->hidden_dim, mlp->output_dim, mlp->batch_size
    );
    printf("output:\n");
    device_to_host_and_print(mlp->batch_size, mlp->output_dim, mlp->output);
    // TODO:
    // softmax
}

// Initialize weights to random values following a normal distribution.
__global__ void random_normal_init_kernel(float* A, int n_elements, unsigned long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n_elements)
    {
        curandState state;
        curand_init(seed, idx, 0, &state);
        A[idx] = curand_normal(&state); 
    } 
}

void random_normal_init(int height, int width, float* A, unsigned long seed)
{
    const int n_elements = height * width;
    const int block_dim = 1024;
    const int grid_dim = (n_elements + block_dim - 1) / block_dim;
    random_normal_init_kernel<<<grid_dim, block_dim>>>(A, n_elements, seed);
}

__global__ void zero_init_kernel(float* A, int n_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n_elements)
    {
        A[idx] = 0.0f;
    }
}

void zero_init(int height, int width, float* A)
{
    const int n_elements = height * width;
    const int block_dim = 1024;
    const int grid_dim = (n_elements + block_dim - 1) / block_dim;
    zero_init_kernel<<<grid_dim, block_dim>>>(A, n_elements);
}

int main()
{
    load_mnist();
    printf("Loaded MNIST\n");
    print_image(train_image[2]);

    constexpr int input_dim = 784;
    constexpr int hidden_dim = 256;
    constexpr int output_dim = 10;
    constexpr int batch_size = 2;
    constexpr int tile_width = 32;

    mlp_t mlp;
    mlp.input_dim = input_dim;
    mlp.hidden_dim = hidden_dim;
    mlp.output_dim = output_dim;
    mlp.batch_size = batch_size;
    CHECK_CUDA(cudaMalloc(&mlp.fc1_w, sizeof(float) * input_dim * hidden_dim));
    CHECK_CUDA(cudaMalloc(&mlp.fc2_w, sizeof(float) * hidden_dim * output_dim));
    CHECK_CUDA(cudaMalloc(&mlp.fc1_b, sizeof(float) * hidden_dim));
    CHECK_CUDA(cudaMalloc(&mlp.fc2_b, sizeof(float) * output_dim));
    CHECK_CUDA(cudaMalloc(&mlp.input, sizeof(float) * batch_size * input_dim));
    CHECK_CUDA(cudaMalloc(&mlp.fc1_w_inter, sizeof(float) * batch_size * hidden_dim));
    CHECK_CUDA(cudaMalloc(&mlp.fc1_b_inter, sizeof(float) * batch_size * hidden_dim));
    CHECK_CUDA(cudaMalloc(&mlp.relu_inter, sizeof(float) * batch_size * hidden_dim));
    CHECK_CUDA(cudaMalloc(&mlp.fc2_w_inter, sizeof(float) * batch_size * output_dim));
    CHECK_CUDA(cudaMalloc(&mlp.fc2_b_inter, sizeof(float) * batch_size * output_dim));
    CHECK_CUDA(cudaMalloc(&mlp.output, sizeof(float) * batch_size * output_dim));

    // Initialize weights and biases.
    random_normal_init(hidden_dim, input_dim, mlp.fc1_w, 0);
    random_normal_init(hidden_dim, output_dim, mlp.fc2_w, 0);
    zero_init(1, hidden_dim, mlp.fc1_b);
    zero_init(1, output_dim, mlp.fc2_b);
    zero_init(batch_size, hidden_dim, mlp.output);
    zero_init(1, output_dim, mlp.output);
    printf("Initialized weights and biases\n");
    //printf("Initial output:\n");
    //device_to_host_and_print(batch_size, output_dim, mlp.output);

    cudaMemcpy(
        mlp.input, &train_image[0], 
        sizeof(float) * input_dim * batch_size, cudaMemcpyHostToDevice
    );
    //device_to_host_and_print(batch_size, input_dim, mlp.input);
    forward_pass<tile_width>(&mlp);
    //printf("First pass output:\n");
    //device_to_host_and_print(batch_size, output_dim, mlp.output);
}
