#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>
#include <curand_kernel.h>

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
    printf("block_dim: %d\ngrid_dim: %d\n", block_dim, grid_dim);
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

template<int tile_width>
__global__ void fc_forward_kernel(
    const float* W, const float* X, float* Y, 
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
    for(int ph = 0; ph < ceil(output_dim/tile_width); ++ph)
    {
        // Load W tile into shared memory.
        if(row < output_dim && ph*tile_width + thread_x < input_dim)
        {
            W_s[thread_y][thread_x] = W[row*input_dim + ph*tile_width + thread_x];
        }
        else
        {
            W_s[thread_y][thread_x] = 0.0f;
        }

        // Load X tile into shared memory.
        if(col < input_dim && ph*tile_width + thread_y < output_dim)
        {
            X_s[thread_y][thread_x] = X[(ph*tile_width + thread_y)*input_dim + col];
        }
        else
        {
            X_s[thread_y][thread_x] = 0.0f;
        }
        __syncthreads();
    
        // Inner loop dot product.
        for(int k = 0; k < tile_width; ++k)
        {
            Y_val += X_s[k][thread_x] * W_s[thread_y][k];
        }
        __syncthreads();
    }

    if(row < output_dim && col < input_dim)
    {
        Y[row*input_dim + col] = Y_val;
    }
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

int main()
{
    constexpr int64_t input_dim = 784;
    constexpr int64_t hidden_dim = 256;
    constexpr int64_t output_dim = 10;
    constexpr int64_t batch_size = 32;

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

    // Initialize weights and biases.
    random_normal_init(hidden_dim, input_dim, d_fc1_weights, 0);
    random_normal_init(hidden_dim, output_dim, d_fc2_weights, 0);
    zero_init(1, hidden_dim, d_fc1_bias);
    zero_init(1, output_dim, d_fc2_bias);
    device_to_host_and_print(output_dim, hidden_dim, d_fc2_weights);

    printf("Hello World\n");
}
