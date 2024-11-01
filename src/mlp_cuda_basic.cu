#include <cuda_runtime.h>
#include <cstdio>

template<int tile_width>
__global__ void fc_forward(
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

int main()
{
    printf("Hello World\n");
}
