#include <cuda.h>
#include <iostream>
#include <cstdio>
#include <cfloat>
#include <cmath>
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

struct  mlp_t
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
    float* dL_dfc2_bias;

    int64_t* labels;
};

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
    printf("softamx back block_x %d\n", block_x);
    softmax_backward_kernel<<<1, block_x>>>(probs, labels, dL_dlogits, n_classes, batch_size);
}

__global__ void bias_out_backward_kernel(
    const float* dL_dlogits, float* dL_dbias, int n_classes, int batch_size
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n_classes * batch_size)
    {
        dL_dbias[idx] = dL_dlogits[idx];
    }
}

void bias_out_backward_launch(const float* dL_dlogits, float* dL_dbias, int n_classes, int batch_size)
{
    const int block_x = ceil((n_classes * batch_size) / (float)32) * 32;
    bias_out_backward_kernel<<<1, block_x>>>(dL_dlogits, dL_dbias, n_classes, batch_size);
}

template<int tile_width, int input_dim, int hidden_dim, int output_dim, int batch_size> 
void forward_pass(mlp_t* mlp)
{
    printf("labels:\n");
    device_to_host_and_print_int(batch_size, 1, mlp->labels);
    fc_forward_launch<tile_width>(
        (const float*)mlp->fc1_w, (const float*)mlp->input, mlp->fc1_w_inter,
        input_dim, hidden_dim, batch_size
    );
    printf("labels:\n");
    device_to_host_and_print_int(batch_size, 1, mlp->labels);

    bias_forward_launch(
        (const float*)mlp->fc1_b, (const float*)mlp->fc1_w_inter, mlp->fc1_b_inter, 
        hidden_dim, batch_size
    );
    relu_forward_launch(
        (const float*)mlp->fc1_w_inter, mlp->relu_inter, hidden_dim, batch_size
    );

    /*printf("fc1_w_inter:\n");
    device_to_host_and_print(batch_size, hidden_dim, mlp->fc1_w_inter);
    printf("\n");
    printf("fc1_b_inter:\n");
    device_to_host_and_print(batch_size, hidden_dim, mlp->fc1_b_inter);
    printf("\n");
    printf("relu_inter:\n");
    device_to_host_and_print(batch_size, hidden_dim, mlp->relu_inter);
    printf("\n");*/

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
    
    printf("fc2_b_inter:\n");
    device_to_host_and_print(batch_size, output_dim, mlp->fc2_b_inter);
    printf("probs:\n");
    device_to_host_and_print(batch_size, output_dim, mlp->probs);
    printf("ce_losses:\n");
    device_to_host_and_print(batch_size, 1, mlp->ce_losses);
    printf("avg_loss:\n");
    device_to_host_and_print(1, 1, mlp->avg_loss); 
}

template<int tile_width, int input_dim, int hidden_dim, int output_dim, int batch_size>
void backward_pass(mlp_t* mlp)
{
    average_backward_launch(mlp->dL_dce, batch_size);
    cross_entropy_backward_launch(
        (const float*)mlp->dL_dce, (const float*)mlp->probs, (const int64_t*)mlp->labels, 
        mlp->dL_dprobs, output_dim, batch_size
    );
    softmax_backward_launch(
        (const float*)mlp->probs, (const int64_t*)mlp->labels, mlp->dL_dlogits, output_dim, batch_size
    );
    bias_out_backward_launch(
        (const float*)mlp->dL_dlogits, mlp->dL_dfc2_bias, output_dim, batch_size
    );

    printf("dL_dce:\n");
    device_to_host_and_print(batch_size, 1, mlp->dL_dce);
    printf("dL_dprobs:\n");
    device_to_host_and_print(batch_size, output_dim, mlp->dL_dprobs);
    printf("dL_dlogits:\n");
    device_to_host_and_print(batch_size, output_dim, mlp->dL_dlogits);
    printf("dL_dfc2_bias:\n");
    device_to_host_and_print(batch_size, output_dim, mlp->dL_dfc2_bias);
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
    constexpr int batch_size = 4;
    constexpr int tile_width = 32;

    mlp_t mlp;
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
    CHECK_CUDA(cudaMalloc(&mlp.probs, sizeof(float) * batch_size * output_dim));
    CHECK_CUDA(cudaMalloc(&mlp.ce_losses, sizeof(float) * batch_size));
    CHECK_CUDA(cudaMalloc(&mlp.labels, sizeof(int64_t) * batch_size));
    CHECK_CUDA(cudaMalloc(&mlp.avg_loss, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp.dL_dce, sizeof(float) * batch_size));
    CHECK_CUDA(cudaMalloc(&mlp.dL_dprobs, sizeof(float) * batch_size * output_dim));
    CHECK_CUDA(cudaMalloc(&mlp.dL_dlogits, sizeof(float) * batch_size * output_dim));
    CHECK_CUDA(cudaMalloc(&mlp.dL_dfc2_bias, sizeof(float) * batch_size * output_dim));

    // Initialize weights and biases.
    random_normal_init(hidden_dim, input_dim, mlp.fc1_w, 0);
    random_normal_init(hidden_dim, output_dim, mlp.fc2_w, 0);
    zero_init(1, hidden_dim, mlp.fc1_b);
    zero_init(1, output_dim, mlp.fc2_b);
    zero_init(1, output_dim, mlp.probs);
    printf("Initialized weights and biases\n");


    int batch_start_idx = 0;
    // Cast batch labels to int64_t.
    int64_t batch_labels[batch_size];
    for(int i = 0; i < batch_size; ++i)
    {
        batch_labels[i] = (int64_t)train_label[batch_start_idx + i];
        printf("%d\n", batch_labels[i]);
    }
    cudaMemcpy(
        mlp.input, &train_image[batch_start_idx], 
        sizeof(float) * input_dim * batch_size, cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        mlp.labels, batch_labels, 
        sizeof(int64_t) * batch_size, cudaMemcpyHostToDevice
    );
    
    forward_pass<tile_width, input_dim, hidden_dim, output_dim, batch_size>(&mlp);
    backward_pass<tile_width, input_dim, hidden_dim, output_dim, batch_size>(&mlp);
    cudaFree(mlp.fc1_w);
    cudaFree(mlp.fc1_b);
    cudaFree(mlp.fc2_w);
    cudaFree(mlp.fc2_b);
    cudaFree(mlp.input);
    cudaFree(mlp.fc1_w_inter);
    cudaFree(mlp.fc1_b_inter);
    cudaFree(mlp.relu_inter);
    cudaFree(mlp.fc2_w_inter);
    cudaFree(mlp.fc2_b_inter);
    cudaFree(mlp.probs);
    cudaFree(mlp.ce_losses);
    cudaFree(mlp.avg_loss);
    cudaFree(mlp.dL_dce);
    cudaFree(mlp.dL_dprobs);
    cudaFree(mlp.dL_dlogits);
}
