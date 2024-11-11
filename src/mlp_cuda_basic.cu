#include <cuda.h>
#include <iostream>
#include <cstdio>
#include <cfloat>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "mnist.hpp"
#include "matrix_utils.hpp"

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

void device_to_host_and_print_int(int height, int width, int64_t* d_A)
{
    size_t mat_size = sizeof(int64_t) * height * width;
    int64_t* h_A = (int64_t*)malloc(mat_size);
    cudaMemcpy(h_A, d_A, mat_size, cudaMemcpyDeviceToHost);
    for(int i = 0; i < height; ++i)
    {
        printf("[");
        for(int k = 0; k < width; ++k)
        {
            printf("%ld ", h_A[i*width + k]);
        }
        printf("]\n");
    }
    free(h_A);
}

void print_average_loss(float* d_losses, int n_losses, int step)
{
    size_t vec_size = sizeof(float) * n_losses;
    float* h_losses = (float*)malloc(vec_size);
    cudaMemcpy(h_losses, d_losses, vec_size, cudaMemcpyDeviceToHost);
    float sum = 0.0f;
    for(int i = 0; i < n_losses; ++i)
    {
        sum += h_losses[i];
    }
    printf("Step %d loss: %f\n", step, sum / (float)n_losses);
}

struct mlp_t
{
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
    float* probs;
    float* ce_losses;
    float* avg_loss;

    float* dL_dce;
    float* dL_dprobs;
    float* dL_dlogits;
    float* dL_dfc2_b;
    float* dL_dfc2_w;
    float* dL_drelu_inter;
    float* dL_dfc1_b_inter;
    float* dL_dfc1_b;
    float* dL_dfc1_w;

    int64_t* labels;
};

void fc_forward_cpu_ref(
    const float* W, // Shape: (input_dim, output_dim)
    const float* X, // Shape: (batch_size, input_dim)
    float* Y,       // Shape: (batch_size, output_dim) 
    int input_dim, int output_dim, int batch_size
){
    for(int row = 0; row < batch_size; ++row)
    {
        for(int col = 0; col < output_dim; ++col)
        {
            float Y_val = 0.0f;
            for(int k = 0; k < input_dim; ++k)
            {
                Y_val += X[row*input_dim + k] * W[k*output_dim + col];
            }
            Y[row*output_dim + col] = Y_val;
        }
    }
}

// TODO: I think there are incorrect memory indices in this kernel and need to double check them.
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
    for(int ph = 0; ph < ceil(input_dim/(float)tile_width); ++ph)
    {
        // Load W tile into shared memory.
        if(col < output_dim && ph*tile_width + thread_y < input_dim)
        {
            // Tiled vertically.
            W_s[thread_y][thread_x] = W[(ph*tile_width + thread_y)*output_dim + col];
        }
        else
        {
            W_s[thread_y][thread_x] = 0.0f;
        }

        // Load X tile into shared memory.
        if(row < batch_size && ph*tile_width + thread_x < input_dim)
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

    if(row < batch_size && col < output_dim)
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

    fc_forward_kernel<tile_width><<<grid_dim, block_dim>>>(W, X, Y, input_dim, output_dim, batch_size);
}

template<int tile_width>
void fc_forward_test(int input_dim, int output_dim, int batch_size, int test_id)
{
    int n_elements_W = input_dim * output_dim;
    int n_elements_X = batch_size * input_dim;
    int n_elements_Y = batch_size * output_dim;

    float* h_W = make_random_matrix(n_elements_W);
    float* h_X = make_random_matrix(n_elements_W);
    float* h_Y = (float*)malloc(sizeof(float) * n_elements_Y);
    float* h_Y_ref = (float*)malloc(sizeof(float) * n_elements_Y);

    float* d_W;
    float* d_X;
    float* d_Y;
    CHECK_CUDA(cudaMalloc(&d_W, sizeof(float) * n_elements_W));
    CHECK_CUDA(cudaMalloc(&d_X, sizeof(float) * n_elements_X));
    CHECK_CUDA(cudaMalloc(&d_Y, sizeof(float) * n_elements_Y));
    cudaMemcpy(d_W, h_W, sizeof(float) * n_elements_W, cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, h_X, sizeof(float) * n_elements_X, cudaMemcpyHostToDevice);

    fc_forward_cpu_ref((const float*)h_W, (const float*)h_X, h_Y_ref, input_dim, output_dim, batch_size);
    fc_forward_launch<tile_width>((const float*)d_W, (const float*)d_X, d_Y, input_dim, output_dim, batch_size);

    cudaMemcpy(h_Y, d_Y, sizeof(float) * n_elements_Y, cudaMemcpyDeviceToHost);

    const float eps = 0.0001f;
    if(!matrices_are_equal(h_Y, h_Y_ref, n_elements_Y, eps))
    {
        printf(
            "fc_forward_test %d failed with input_dim = %d, output_dim = %d, batch_size = %d\n", 
            test_id, input_dim, output_dim, batch_size
        );
    }
    else
    {
        printf("fc_forward_test %d passed\n", test_id);
    }

    cudaFree(d_W);
    cudaFree(d_X);
    cudaFree(d_Y);
    free(h_W);
    free(h_X);
    free(h_Y);
    free(h_Y_ref);
}

void bias_forward_cpu_ref(const float* B, const float* X, float* Y, int input_dim, int batch_size)
{
    for(int row = 0; row < batch_size; ++row)
    {
        for(int col = 0; col < input_dim; ++col)
        {
            Y[row*input_dim + col] = X[row*input_dim + col] + B[col];
        }
    }
}

__global__ void bias_forward_kernel(const float* B, const float* X, float* Y, int input_dim, int batch_size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(col < input_dim && row < batch_size)
    {
        int idx = row*input_dim + col;
        Y[row*input_dim + col] = X[idx] + B[col];
    }
}

void bias_forward_launch(const float* B, const float* X, float* Y, int input_dim, int batch_size)
{
    const int block_x = (input_dim + 31) / 32;
    dim3 grid_dim((input_dim + block_x - 1) / block_x, batch_size);
    dim3 block_dim(block_x, 1);
    bias_forward_kernel<<<grid_dim, block_dim>>>(B, X, Y, input_dim, batch_size);
}

void bias_forward_test(int input_dim, int batch_size, int test_id)
{
    int n_elements_B = input_dim;
    int n_elements_X = batch_size * input_dim;
    int n_elements_Y = n_elements_X;

    float* h_B = make_random_matrix(n_elements_B);
    float* h_X = make_random_matrix(n_elements_X);
    float* h_Y = (float*)malloc(sizeof(float) * n_elements_Y);
    float* h_Y_ref = (float*)malloc(sizeof(float) * n_elements_Y);

    float* d_B;
    float* d_X;
    float* d_Y;
    CHECK_CUDA(cudaMalloc(&d_B, sizeof(float) * n_elements_B));
    CHECK_CUDA(cudaMalloc(&d_X, sizeof(float) * n_elements_X));
    CHECK_CUDA(cudaMalloc(&d_Y, sizeof(float) * n_elements_Y));
    cudaMemcpy(d_B, h_B, sizeof(float) * n_elements_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, h_X, sizeof(float) * n_elements_X, cudaMemcpyHostToDevice);

    bias_forward_cpu_ref((const float*)h_B, (const float*)h_X, h_Y_ref, input_dim, batch_size);
    bias_forward_launch((const float*)d_B, (const float*)d_X, d_Y, input_dim, batch_size);

    cudaMemcpy(h_Y, d_Y, sizeof(float) * n_elements_Y, cudaMemcpyDeviceToHost);
    
    const float eps = 0.0001f;
    if(!matrices_are_equal(h_Y, h_Y_ref, n_elements_Y, eps))
    {
        printf(
            "bias_forward_test %d failed with input_dim = %d, batch_size = %d\n", 
            test_id, input_dim, batch_size
        );
    }
    else
    {
        printf("bias_forward_test %d passed\n", test_id);
    }

    cudaFree(d_B);
    cudaFree(d_X);
    cudaFree(d_Y);
    free(h_B);
    free(h_X);
    free(h_Y);
    free(h_Y_ref);
}

void relu_forward_cpu_ref(const float* X, float* Y, int input_dim, int batch_size)
{
    for(int row = 0; row < batch_size; ++row)
    {
        for(int col = 0; col < input_dim; ++col)
        {
            float X_val = X[row*input_dim + col];
            Y[row*input_dim + col] = (X_val > 0.0f) ? X_val : 0.0f;
        }
    }
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
    const int block_x = ceil(input_dim / (float)32) * 32;
    dim3 grid_dim(1, batch_size);
    dim3 block_dim(block_x, 1);
    relu_forward_kernel<<<grid_dim, block_dim>>>(X, Y, input_dim, batch_size);
}

void relu_forward_test(int input_dim, int batch_size, int test_id)
{
    int n_elements_X = batch_size * input_dim;
    int n_elements_Y = n_elements_X;

    float* h_X = make_random_matrix(n_elements_X);
    float* h_Y = (float*)malloc(sizeof(float) * n_elements_Y);
    float* h_Y_ref = (float*)malloc(sizeof(float) * n_elements_Y);

    float* d_X;
    float* d_Y;
    CHECK_CUDA(cudaMalloc(&d_X, sizeof(float) * n_elements_X));
    CHECK_CUDA(cudaMalloc(&d_Y, sizeof(float) * n_elements_Y));
    cudaMemcpy(d_X, h_X, sizeof(float) * n_elements_X, cudaMemcpyHostToDevice);

    relu_forward_cpu_ref((const float*)h_X, h_Y_ref, input_dim, batch_size);
    relu_forward_launch((const float*)d_X, d_Y, input_dim, batch_size);

    cudaMemcpy(h_Y, d_Y, sizeof(float) * n_elements_Y, cudaMemcpyDeviceToHost);
    
    const float eps = 0.0001f;
    if(!matrices_are_equal(h_Y, h_Y_ref, n_elements_Y, eps))
    {
        printf(
            "relu_forward_test %d failed with input_dim = %d, batch_size = %d\n", 
            test_id, input_dim, batch_size
        );
    }
    else
    {
        printf("relu_forward_test %d passed\n", test_id);
    }

    cudaFree(d_X);
    cudaFree(d_Y);
    free(h_X);
    free(h_Y);
    free(h_Y_ref);
}

void softmax_forward_cpu_ref(const float* X, float* Y, int input_dim, int batch_size)
{
    for(int row = 0; row < batch_size; ++row)
    {
        // Find each row's maxmimum value. 
        float row_max = -FLT_MAX;
        for(int col = 0; col < input_dim; ++col)
        {
            float X_val = X[row*input_dim + col];
            if(row_max < X_val)
            {
                row_max = X_val;
            }
        }

        // Calculate the row's sum of exponentials.
        float row_exp_sum = 0.0f;
        for(int col = 0; col < input_dim; ++col)
        {
            row_exp_sum += expf(X[row*input_dim + col] - row_max);
        }

        // Calculate final output.
        for(int col = 0; col < input_dim; ++col)
        {
            Y[row*input_dim + col] = expf(X[row*input_dim + col] - row_max) / row_exp_sum;
        }
    }
}

template<int rows_per_block, int input_dim, int batch_size>
__global__ void softmax_forward_kernel(const float* X, float* Y)
{
    constexpr int elements_per_block = rows_per_block * input_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = idx / input_dim;

    __shared__ float X_s[elements_per_block];
    __shared__ float row_max[rows_per_block];
    __shared__ float row_exp_sum[rows_per_block];
    
    if(idx < input_dim * batch_size && threadIdx.x < elements_per_block)
    {
        X_s[threadIdx.x] = X[idx];
    }
    __syncthreads();

    // Turn off all threads except for the ones mapped to the start of each row.
    if(idx % input_dim == 0 && row_idx < batch_size)
    {
        // Find each row's maxmimum value. 
        row_max[row_idx] = -FLT_MAX;
        for(int i = 0; i < input_dim; ++i)
        {
            float X_val = X_s[row_idx * input_dim + i];
            if(row_max[row_idx] < X_val)
            {
                row_max[row_idx] = X_val;
            }
        }
        
        // Calculate the row's sum of exponentials.
        row_exp_sum[row_idx] = 0.0f;
        for(int i = 0; i < input_dim; ++i)
        {
            row_exp_sum[row_idx] += __expf(X_s[row_idx * input_dim + i] - row_max[row_idx]);
        }
    }
    __syncthreads();

    if(idx < input_dim * batch_size && threadIdx.x < elements_per_block)
    {
        Y[idx] = __expf(X_s[threadIdx.x] - row_max[row_idx]) / row_exp_sum[row_idx];
    }
}

template<int input_dim, int batch_size>
void softmax_forward_launch(const float* X, float* Y)
{
    constexpr int rows_per_block = (6 > batch_size) ? batch_size: 6;
    dim3 grid_dim((batch_size + rows_per_block - 1) / rows_per_block);
    dim3 block_dim(ceil((rows_per_block * input_dim) / (float)32) * 32);
    softmax_forward_kernel<rows_per_block, input_dim, batch_size><<<grid_dim, block_dim>>>(X, Y);
}

template<int input_dim, int batch_size>
void softmax_forward_test(int test_id)
{
    int n_elements_X = batch_size * input_dim;
    int n_elements_Y = n_elements_X;

    float* h_X = make_random_matrix(n_elements_X);
    float* h_Y = (float*)malloc(sizeof(float) * n_elements_Y);
    float* h_Y_ref = (float*)malloc(sizeof(float) * n_elements_Y);

    float* d_X;
    float* d_Y;
    CHECK_CUDA(cudaMalloc(&d_X, sizeof(float) * n_elements_X));
    CHECK_CUDA(cudaMalloc(&d_Y, sizeof(float) * n_elements_Y));
    cudaMemcpy(d_X, h_X, sizeof(float) * n_elements_X, cudaMemcpyHostToDevice);

    softmax_forward_cpu_ref((const float*)h_X, h_Y_ref, input_dim, batch_size);
    softmax_forward_launch<input_dim, batch_size>((const float*)d_X, d_Y);

    cudaMemcpy(h_Y, d_Y, sizeof(float) * n_elements_Y, cudaMemcpyDeviceToHost);
    
    const float eps = 0.0001f;
    if(!matrices_are_equal(h_Y, h_Y_ref, n_elements_Y, eps))
    {
        printf(
            "softmax_forward_test %d failed with input_dim = %d, batch_size = %d\n", 
            test_id, input_dim, batch_size
        );
    }
    else
    {
        printf("softmax_forward_test %d passed\n", test_id);
    }

    cudaFree(d_X);
    cudaFree(d_Y);
    free(h_X);
    free(h_Y);
    free(h_Y_ref);
}

void cross_entropy_forward_cpu_ref(const float* X, const int64_t* T, float* Y, int n_classes, int batch_size)
{
    constexpr float eps = 0.00001f;
    for(int row = 0; row < batch_size; ++row)
    {
        Y[row] = -1.0f * logf(X[row*n_classes + T[row]] + eps);
    }
}

__global__ void cross_entropy_forward_kernel(
    const float* X, const int64_t* T, float* Y, int n_classes, int batch_size
){
    constexpr float eps = 0.00001f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < batch_size)
    {
        Y[idx] = -1.0f * __logf(X[idx * n_classes + T[idx]] + eps);
    }
}

void cross_entropy_forward_launch(const float* X, const int64_t* T, float* Y, int n_classes, int batch_size)
{
    dim3 grid_dim(1);
    dim3 block_dim(ceil(batch_size / (float)32) * 32);
    cross_entropy_forward_kernel<<<grid_dim, block_dim>>>(X, T, Y, n_classes, batch_size);
}

void cross_entropy_forward_test(int n_classes, int batch_size, int test_id)
{
    int n_elements_X = batch_size * n_classes;
    int n_elements_T = batch_size;
    int n_elements_Y = batch_size;

    float* h_X = make_random_matrix(n_elements_X);
    int64_t* h_T = make_random_labels(n_elements_T);
    float* h_Y = (float*)malloc(sizeof(float) * n_elements_Y);
    float* h_Y_ref = (float*)malloc(sizeof(float) * n_elements_Y);

    float* d_X;
    int64_t* d_T;
    float* d_Y;
    CHECK_CUDA(cudaMalloc(&d_X, sizeof(float) * n_elements_X));
    CHECK_CUDA(cudaMalloc(&d_T, sizeof(int64_t) * n_elements_T));
    CHECK_CUDA(cudaMalloc(&d_Y, sizeof(float) * n_elements_Y));
    cudaMemcpy(d_X, h_X, sizeof(float) * n_elements_X, cudaMemcpyHostToDevice);
    cudaMemcpy(d_T, h_T, sizeof(int64_t) * n_elements_T, cudaMemcpyHostToDevice);

    cross_entropy_forward_cpu_ref((const float*)h_X, (const int64_t*)h_T, h_Y_ref, n_classes, batch_size);
    cross_entropy_forward_launch((const float*)d_X, (const int64_t*)d_T, d_Y, n_classes, batch_size);
    
    cudaMemcpy(h_Y, d_Y, sizeof(float) * n_elements_Y, cudaMemcpyDeviceToHost);
    
    const float eps = 0.0001f;
    if(!matrices_are_equal(h_Y, h_Y_ref, n_elements_Y, eps))
    {
        printf(
            "cross_entropy_forward_test %d failed with n_classes = %d, batch_size = %d\n", 
            test_id, n_classes, batch_size
        );
    }
    else
    {
        printf("cross_entropy_forward_test %d passed\n", test_id);
    }
    return;

    cudaFree(d_X);
    cudaFree(d_T);
    cudaFree(d_Y);
    free(h_X);
    free(h_T);
    free(h_Y);
    free(h_Y_ref);
}

__global__ void average_forward_kernel(const float* X, float* Y, int n_inputs)
{
    if(threadIdx.x == 0 && blockIdx.x ==0)
    {
        // Sequential sum for simplicity because we're expecting a small number of inputs.
        float sum = 0.0f;
        for(int i = 0; i < n_inputs; ++i)
        {
            sum += X[i];
        }
        *Y = sum / (float)n_inputs;
    }
}

void average_forward_launch(const float* X, float* Y, int n_inputs)
{
    // We have to launch a whole warp, but we're only using one thread.
    average_forward_kernel<<<1, 32>>>(X, Y, n_inputs);
}

__global__ void average_backward_kernel(float* dL_dX, int n_inputs)
{
    if(threadIdx.x == 0 && blockIdx.x == 0)
    {
        float partial_deriv = 1.0f / (float)n_inputs;
        for(int i = 0; i < n_inputs; ++i)
        {
            dL_dX[i] = partial_deriv;
        }
    }
}

void average_backward_launch(float* dL_dX, int n_inputs)
{
    // We have to launch a whole warp, but we're only using one thread.
    average_backward_kernel<<<1, 32>>>(dL_dX, n_inputs);
}

// NOTE: not using dL_dce right now since it seems like a pretty simple kernel fusion.
// Maybe remove the average_backward_kernel.
__global__ void cross_entropy_backward_kernel(
    const float* dL_dce, const float* probs, const int64_t* labels, 
    float* dL_dprobs, int n_classes, int batch_size
){
    constexpr float eps = 0.000001f;
    constexpr float max_grad = 30.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n_classes * batch_size)
    {
        int label = labels[idx / n_classes];
        float val = 0.0f;
        if(idx % n_classes == label)
        {
            float prob = fmaxf(probs[idx], eps);
            val = -1.0f / ((float)batch_size * prob);
            val = fmaxf(fminf(val, max_grad), -max_grad);
        }
        dL_dprobs[idx] = val;
    }
}

void cross_entropy_backward_launch(
    const float* dL_dce, const float* probs, const int64_t* labels, 
    float* dL_dprobs, int n_classes, int batch_size
){
    const int block_x = ceil((n_classes * batch_size) / (float)32) * 32;
    //printf("block x %d\n", block_x);
    cross_entropy_backward_kernel<<<1, block_x>>>(dL_dce, probs, labels, dL_dprobs, n_classes, batch_size);
}

__global__ void softmax_backward_kernel(
    const float* probs, const int64_t* labels, float* dL_dlogits, int n_classes, int batch_size
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n_classes * batch_size)
    {
        float label_term = (labels[idx / n_classes] == idx % n_classes) ? 1.0f : 0.0f;
        dL_dlogits[idx] = probs[idx] - label_term;
    }
}

void softmax_backward_launch(
    const float* probs, const int64_t* labels, float* dL_dlogits, int n_classes, int batch_size
){
    const int block_x = ceil((n_classes * batch_size) / (float)32) * 32;
    softmax_backward_kernel<<<1, block_x>>>(probs, labels, dL_dlogits, n_classes, batch_size);
}

__global__ void bias_backward_kernel(
    const float* dL_dlogits, float* dL_dbias, int n_classes, int batch_size
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n_classes * batch_size)
    {
        dL_dbias[idx] = dL_dlogits[idx];
    }
}

void bias_backward_launch(const float* dL_dlogits, float* dL_dbias, int n_classes, int batch_size)
{
    const int block_x = ceil((n_classes * batch_size) / (float)32) * 32;
    bias_backward_kernel<<<1, block_x>>>(dL_dlogits, dL_dbias, n_classes, batch_size);
}

__global__ void fc_backward_w_kernel(
    const float* dL_dY, const float* X, float* dL_dW, int input_dim, int output_dim, int batch_size
){
    int batch = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / input_dim;
    int col = idx % input_dim;

    if(batch < batch_size && col < input_dim && row < output_dim)
    {
        dL_dW[batch*input_dim*output_dim + row*input_dim + col] = 
            dL_dY[batch*output_dim + row] * X[batch*input_dim + col];
    }
}

void fc_backward_w_launch(
    const float* dL_dY, const float* X, float* dL_dW, 
    int input_dim, int output_dim, int batch_size
){
    const int threads_per_sample = ceil((input_dim * output_dim) / (float)32) * 32;
    const int n_blocks = ceil(threads_per_sample / (float)1024);
    dim3 block_dim(ceil(threads_per_sample / (float)n_blocks), 1, 1);
    dim3 grid_dim(n_blocks, batch_size);
    fc_backward_w_kernel<<<grid_dim, block_dim>>>(dL_dY, X, dL_dW, input_dim, output_dim, batch_size);

    /*printf("input_dim %d\n", input_dim);
    printf("output_dim %d\n", output_dim);
    printf("threads_per_sample %d\n", threads_per_sample);
    printf("n_blocks %d\n", n_blocks);
    printf("block_dim.x %d\n", block_dim.x);
    printf("block_dim.y %d\n", block_dim.y);
    printf("grid_dim.x %d\n", grid_dim.x);
    printf("grid_dim.y %d\n", grid_dim.y);*/
}

__global__ void fc_backward_x_kernel(
    const float* dL_dY, const float* W, float* dL_dX, 
    int input_dim, int output_dim, int batch_size
){
    int batch = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < input_dim && batch < batch_size)
    {
        float val = 0.0f;
        for(int k = 0; k < output_dim; ++k)
        {
            val += dL_dY[batch*output_dim + k] * W[k*input_dim + idx];
        }
        dL_dX[batch*input_dim + idx] = val;
    }
}

void fc_backward_x_launch(
    const float* dL_dY, const float* X, float* dL_dX,
    int input_dim, int output_dim, int batch_size
){
    dim3 block_dim(ceil(input_dim / (float)32) * 32, 1);
    dim3 grid_dim(1, batch_size);
    fc_backward_x_kernel<<<grid_dim, block_dim>>>(dL_dY, X, dL_dX, input_dim, output_dim, batch_size);

    /*printf("block_dim.x %d\n", block_dim.x);
    printf("block_dim.y %d\n", block_dim.y);
    printf("grid_dim.x %d\n", grid_dim.x);
    printf("grid_dim.y %d\n", grid_dim.y);*/
}

__global__ void relu_backward_kernel(
    const float* dL_dY, const float* X, float* dL_dX, int input_dim, int batch_size
){
    int batch = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(batch < batch_size && idx < input_dim)
    {
        float X_val = X[batch*input_dim + idx];
        float dL_dY_val = dL_dY[batch*input_dim + idx];
        dL_dX[batch*input_dim + idx] = ((X_val > 0.0f) ? 1.0f : 0.0f) * dL_dY_val;
    }
}

void relu_backward_launch(
    const float* dL_dY, const float* X, float* dL_dX, int input_dim, int batch_size
){
    const int block_x = ceil(input_dim / (float)32) * 32;
    dim3 grid_dim(1, batch_size);
    dim3 block_dim(block_x, 1);
    relu_backward_kernel<<<grid_dim, block_dim>>>(dL_dY, X, dL_dX, input_dim, batch_size);
}

__global__ void gradient_descent_kernel(
    float* grad, float* params, float learning_rate, int n_elements, int batch_size
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n_elements)
    {
        float grad_sum = 0.0f;
        for(int batch = 0; batch < batch_size; ++batch)
        {
            grad_sum += grad[batch*n_elements + idx];
        }
        params[idx] -= learning_rate * grad_sum;
    }
}

void gradient_descent_launch(float* grad, float* params, float learning_rate, int n_elements, int batch_size)
{
    const int total_threads = ceil(n_elements / (float)32) * 32;
    const int n_blocks = ceil(total_threads / (float)1024);
    dim3 block_dim(ceil(total_threads / (float)n_blocks));
    dim3 grid_dim(n_blocks);
    gradient_descent_kernel<<<grid_dim, block_dim>>>(grad, params, learning_rate, n_elements, batch_size);
    
    /*printf("total_threads %d\n", total_threads);
    printf("n_blocks %d\n", n_blocks);
    printf("block_dim.x %d\n", block_dim.x);
    printf("grid_dim.x %d\n", grid_dim.x);*/
}

template<int tile_width, int input_dim, int hidden_dim, int output_dim, int batch_size>
void test_kernels()
{
    printf("Testing kernels...\n");
    
    fc_forward_test<tile_width>(256, 256, 32, 0);
    fc_forward_test<tile_width>(input_dim, hidden_dim, batch_size, 1);
    fc_forward_test<tile_width>(hidden_dim, output_dim, batch_size, 2);

    bias_forward_test(256, 32, 0);
    bias_forward_test(input_dim, batch_size, 1);
    bias_forward_test(hidden_dim, batch_size, 2);
    
    relu_forward_test(256, 32, 0);
    relu_forward_test(hidden_dim, batch_size, 1);
    relu_forward_test(output_dim, batch_size, 2);

    //softmax_forward_test<256, 32>(0);
    softmax_forward_test<output_dim, batch_size>(1);

    cross_entropy_forward_test(10, 4, 0);
}

template<int tile_width, int input_dim, int hidden_dim, int output_dim, int batch_size> 
void forward_pass(mlp_t* mlp)
{
    fc_forward_launch<tile_width>(
        (const float*)mlp->fc1_w, (const float*)mlp->input, mlp->fc1_w_inter,
        input_dim, hidden_dim, batch_size
    );
    bias_forward_launch(
        (const float*)mlp->fc1_b, (const float*)mlp->fc1_w_inter, mlp->fc1_b_inter, 
        hidden_dim, batch_size
    );
    relu_forward_launch(
        (const float*)mlp->fc1_w_inter, mlp->relu_inter, hidden_dim, batch_size
    );

    fc_forward_launch<tile_width>(
        (const float*)mlp->fc2_w, (const float*)mlp->relu_inter, mlp->fc2_w_inter,
        hidden_dim, output_dim, batch_size
    );
    bias_forward_launch(
        (const float*)mlp->fc2_b, (const float*)mlp->fc2_w_inter, mlp->fc2_b_inter, 
        output_dim, batch_size
    );
    softmax_forward_launch<output_dim, batch_size>((const float*)mlp->fc2_b_inter, mlp->probs);
    cross_entropy_forward_launch(
        (const float*)mlp->probs, (const int64_t*)mlp->labels, mlp->ce_losses, 10, batch_size
    );
    average_forward_launch((const float*)mlp->ce_losses, mlp->avg_loss, batch_size);
}

template<int tile_width, int input_dim, int hidden_dim, int output_dim, int batch_size>
void backward_pass(mlp_t* mlp, float learning_rate)
{
    average_backward_launch(mlp->dL_dce, batch_size);
    cross_entropy_backward_launch(
        (const float*)mlp->dL_dce, (const float*)mlp->probs, (const int64_t*)mlp->labels, 
        mlp->dL_dprobs, output_dim, batch_size
    );
    softmax_backward_launch(
        (const float*)mlp->probs, (const int64_t*)mlp->labels, mlp->dL_dlogits, output_dim, batch_size
    );
    bias_backward_launch(
        (const float*)mlp->dL_dlogits, mlp->dL_dfc2_b, output_dim, batch_size
    );
    fc_backward_w_launch(
        (const float*)mlp->dL_dfc2_b, (const float*)mlp->relu_inter, mlp->dL_dfc2_w, 
        hidden_dim, output_dim, batch_size
    );
    fc_backward_x_launch(
        (const float*)mlp->dL_dlogits, (const float*)mlp->fc2_w, mlp->dL_drelu_inter, 
        hidden_dim, output_dim, batch_size
    );
    relu_backward_launch(
        (const float*)mlp->dL_drelu_inter, (const float*)mlp->fc1_b_inter, 
        mlp->dL_dfc1_b_inter, hidden_dim, batch_size
    );
    bias_backward_launch(
        (const float*)mlp->dL_dfc1_b_inter, mlp->dL_dfc1_b, hidden_dim, batch_size
    );
    fc_backward_w_launch(
        (const float*)mlp->dL_dfc1_b, (const float*)mlp->input, 
        mlp->dL_dfc1_w, input_dim, hidden_dim, batch_size
    );
    
    gradient_descent_launch(mlp->dL_dfc2_b, mlp->fc2_b, learning_rate, output_dim, batch_size);
    gradient_descent_launch(mlp->dL_dfc2_w, mlp->fc2_w, learning_rate, hidden_dim*output_dim, batch_size);
    gradient_descent_launch(mlp->dL_dfc1_b, mlp->fc1_b, learning_rate, hidden_dim, batch_size);
    gradient_descent_launch(mlp->dL_dfc1_w, mlp->fc1_w, learning_rate, input_dim*hidden_dim, batch_size);
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

void mlp_init(mlp_t* mlp, int input_dim, int hidden_dim, int output_dim, int batch_size)
{
    CHECK_CUDA(cudaMalloc(&mlp->fc1_w, sizeof(float) * input_dim * hidden_dim));
    CHECK_CUDA(cudaMalloc(&mlp->fc2_w, sizeof(float) * hidden_dim * output_dim));
    CHECK_CUDA(cudaMalloc(&mlp->fc1_b, sizeof(float) * hidden_dim));
    CHECK_CUDA(cudaMalloc(&mlp->fc2_b, sizeof(float) * output_dim));
    CHECK_CUDA(cudaMalloc(&mlp->input, sizeof(float) * batch_size * input_dim));
    CHECK_CUDA(cudaMalloc(&mlp->fc1_w_inter, sizeof(float) * batch_size * hidden_dim));
    CHECK_CUDA(cudaMalloc(&mlp->fc1_b_inter, sizeof(float) * batch_size * hidden_dim));
    CHECK_CUDA(cudaMalloc(&mlp->relu_inter, sizeof(float) * batch_size * hidden_dim));
    CHECK_CUDA(cudaMalloc(&mlp->fc2_w_inter, sizeof(float) * batch_size * output_dim));
    CHECK_CUDA(cudaMalloc(&mlp->fc2_b_inter, sizeof(float) * batch_size * output_dim));
    CHECK_CUDA(cudaMalloc(&mlp->probs, sizeof(float) * batch_size * output_dim));
    CHECK_CUDA(cudaMalloc(&mlp->ce_losses, sizeof(float) * batch_size));
    CHECK_CUDA(cudaMalloc(&mlp->labels, sizeof(int64_t) * batch_size));
    CHECK_CUDA(cudaMalloc(&mlp->avg_loss, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->dL_dce, sizeof(float) * batch_size));
    CHECK_CUDA(cudaMalloc(&mlp->dL_dprobs, sizeof(float) * batch_size * output_dim));
    CHECK_CUDA(cudaMalloc(&mlp->dL_dlogits, sizeof(float) * batch_size * output_dim));
    CHECK_CUDA(cudaMalloc(&mlp->dL_dfc2_b, sizeof(float) * batch_size * output_dim));
    CHECK_CUDA(cudaMalloc(&mlp->dL_dfc2_w, sizeof(float) * batch_size * hidden_dim * output_dim));
    CHECK_CUDA(cudaMalloc(&mlp->dL_drelu_inter, sizeof(float) * batch_size * hidden_dim * output_dim));
    CHECK_CUDA(cudaMalloc(&mlp->dL_dfc1_b_inter, sizeof(float) * batch_size * hidden_dim));
    CHECK_CUDA(cudaMalloc(&mlp->dL_dfc1_b, sizeof(float) * batch_size * hidden_dim));
    CHECK_CUDA(cudaMalloc(&mlp->dL_dfc1_w, sizeof(float) * batch_size * hidden_dim * input_dim));

    // Initialize weights and biases.
    random_normal_init(hidden_dim, input_dim, mlp->fc1_w, 0);
    random_normal_init(hidden_dim, output_dim, mlp->fc2_w, 0);
    zero_init(1, hidden_dim, mlp->fc1_b);
    zero_init(1, output_dim, mlp->fc2_b);
    zero_init(1, output_dim, mlp->probs);
}

void mlp_free(mlp_t* mlp)
{
    cudaFree(mlp->fc1_w);
    cudaFree(mlp->fc1_b);
    cudaFree(mlp->fc2_w);
    cudaFree(mlp->fc2_b);
    cudaFree(mlp->input);
    cudaFree(mlp->fc1_w_inter);
    cudaFree(mlp->fc1_b_inter);
    cudaFree(mlp->relu_inter);
    cudaFree(mlp->fc2_w_inter);
    cudaFree(mlp->fc2_b_inter);
    cudaFree(mlp->probs);
    cudaFree(mlp->ce_losses);
    cudaFree(mlp->avg_loss);
    cudaFree(mlp->dL_dce);
    cudaFree(mlp->dL_dprobs);
    cudaFree(mlp->dL_dlogits);
    cudaFree(mlp->dL_dfc2_b);
    cudaFree(mlp->dL_dfc2_w);
    cudaFree(mlp->dL_drelu_inter);
    cudaFree(mlp->dL_dfc1_b_inter);
    cudaFree(mlp->dL_dfc1_b);
    cudaFree(mlp->dL_dfc1_w);
}

int main()
{
    load_mnist();
    printf("Loaded MNIST\n");
    print_image(train_image[2]);

    constexpr int input_dim = 784;
    constexpr int hidden_dim = 256;
    constexpr int output_dim = 10;
    constexpr int batch_size = 4;
    constexpr int tile_width = 32;
    constexpr float learning_rate = 0.0001f;
    test_kernels<tile_width, input_dim, hidden_dim, output_dim, batch_size>();

    mlp_t mlp;
    mlp_init(&mlp, input_dim, hidden_dim, output_dim, batch_size);
    printf("Initialized weights and biases\n");

    int batch_start_idx = 0;
    // Cast batch labels to int64_t.
    int64_t batch_labels[batch_size];
    for(int i = 0; i < batch_size; ++i)
    {
        batch_labels[i] = (int64_t)train_label[batch_start_idx + i];
        //printf("%d\n", batch_labels[i]);
    }
    cudaMemcpy(
        mlp.input, &train_image[batch_start_idx], 
        sizeof(float) * input_dim * batch_size, cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        mlp.labels, batch_labels, 
        sizeof(int64_t) * batch_size, cudaMemcpyHostToDevice
    );
    
    for(int i = 0; i < 30000; ++i)
    {
        forward_pass<tile_width, input_dim, hidden_dim, output_dim, batch_size>(&mlp);
        backward_pass<tile_width, input_dim, hidden_dim, output_dim, batch_size>(&mlp, learning_rate);
        if(i % 1000 == 0)
        {
            print_average_loss(mlp.ce_losses, batch_size, i);
        }
    }

    mlp_free(&mlp);
}
