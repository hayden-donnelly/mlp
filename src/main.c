#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "mnist.h"

#define INPUT_DIM 784 // 28*28
#define NUM_HIDDEN_LAYERS 3
#define HIDDEN_DIM 256
#define OUTPUT_DIM 10
#define BATCH_SIZE 1

const int input_layer_param_count = INPUT_DIM * HIDDEN_DIM;
const int hidden_layer_param_count = HIDDEN_DIM * HIDDEN_DIM;
const int output_layer_param_count = HIDDEN_DIM * OUTPUT_DIM;
 
const double two_pi = 2.0*3.14159265358979323846;

float random_normal()
{
    const double epsilon = DBL_MIN;
    static float z1, z2;
    static int generate;
    
    generate = !generate;
    if(!generate)
    {
        return z2;
    }

    double u1, u2;
    do
    {
       u1 = rand() * (1.0 / RAND_MAX);
       u2 = rand() * (1.0 / RAND_MAX);
    }
    while (u1 <= epsilon);

    z1 = (float)(sqrt(-2.0 * log(u1)) * cos(two_pi * u2));
    z2 = (float)(sqrt(-2.0 * log(u1)) * sin(two_pi * u2));
    return z1;
}

float* init_params()
{
    const int total_param_count = 
        (NUM_HIDDEN_LAYERS * hidden_layer_param_count)
        + input_layer_param_count + output_layer_param_count;

    float* params = (float*)malloc(sizeof(float) * total_param_count);
    for(int i = 0; i < total_param_count; i++)
    {
        params[i] = random_normal();
    }
    return params;
}

static inline void scalar_relu(float* a)
{
    *a = (*a > 0.0f) ? *a : 0.0f;
}

static inline void sigmoid(float *a)
{
    *a = 1.0f / (1.0f + (float)exp(-(double)*a));
}

static inline void vec_mat_mul_sigmoid(
    float* mat, float* vec_in, float* vec_out, 
    const int in_dim, const int out_dim
){
    int mat_offset = 0;
    for(int out_idx = 0; out_idx < out_dim; out_idx++)
    {
        vec_out[out_idx] = 0.0f;
        for(int in_idx = 0; in_idx < in_dim; in_idx++)
        {
            vec_out[out_idx] += mat[mat_offset++] * vec_in[in_idx];
        }
        sigmoid(vec_out + out_idx);
    }
}

static inline void vec_mat_mul_sigmoid_backward(
    float* mat, float* vec_in, 
    float* vec_out, float* vec_out_grad,
    float* mat_grad, float* vec_in_grad,
    const int in_dim, const int out_dim
){
    // Notation: f: R^(vec_in) -> R^(vec_out), f(x) = Wx.
    float vec_mat_mul_grad[out_dim];
    for(int i = 0; i < out_dim; i++)
    {
        // dsig_i/df_i = (sig(f_i) * (1 - sig(f_i)).
        // dL/df_i = dsig_i/df_i * dL/dsig_i.
        vec_mat_mul_grad[i] = (vec_out[i] * (1.0f - vec_out[i])) * vec_out_grad[i];
    }
    for(int in_idx = 0; in_idx < in_dim; in_idx++)
    {
        for(int out_idx = 0; out_idx < out_dim; out_idx++)
        {
            // df_j/dx_i = W_ij.
            // dL/dx_i = [df_1/dx_i * dL/df_1, df_2/dx_i * dL/df_2, ... , df_n/dx_i * dL/df_n.
            // Take a transposed view of the matrix.
            const int mat_offset = out_idx * in_dim;
            vec_in_grad[in_idx] += mat[mat_offset] * vec_mat_mul_grad[out_idx];
        }
    }
    int mat_grad_offset = 0;
    for(int out_idx = 0; out_idx < out_dim; out_idx++)
    {
        for(int in_idx = 0; in_idx < in_dim; in_idx++)
        {
            // dL/dW = dL/df * x^T.
            // dL/dW_ij = x_j * dL/df_i.
            mat_grad[mat_grad_offset++] = vec_in[in_idx] * vec_mat_mul_grad[out_idx];
        }
    }
}

static inline void vec_mat_mul_softmax(
    float* mat, float* vec_in, float* vec_out, 
    const int in_dim, const int out_dim
){
    int mat_offset = 0;
    double exp_layer_sum = 0.0f;
    for(int out_idx = 0; out_idx < out_dim; out_idx++)
    {
        vec_out[out_idx] = 0.0f;
        for(int in_idx = 0; in_idx < in_dim; in_idx++)
        {
            vec_out[out_idx] += mat[mat_offset++] * vec_in[in_idx];
        }
        exp_layer_sum += exp((double)vec_out[out_idx]);
    }
    for(int i = 0; i < out_dim; i++)
    {
        vec_out[i] = (float)(exp((double)vec_out[i]) / exp_layer_sum); 
    }
}

void print_output(float* out, const int n)
{
    for(int i = 0; i < n; i++)
    {
        printf("%f ", out[i]);
    }
    printf("\n");
}

void forward_pass(float* params, float* in, float* out, float* activations)
{
    vec_mat_mul_sigmoid(params, in, activations, INPUT_DIM, HIDDEN_DIM);
    printf("Input layer:\n");
    print_output(activations, HIDDEN_DIM);
    int activations_offset = 0;
    int params_offset = input_layer_param_count;
    for(int i = 0; i < NUM_HIDDEN_LAYERS; i++)
    {
        const int next_activations_offset = activations_offset + HIDDEN_DIM;
        vec_mat_mul_sigmoid(
            params + params_offset, 
            activations + activations_offset, 
            activations + next_activations_offset,
            HIDDEN_DIM,
            HIDDEN_DIM
        );
        printf("Hidden layer %d:\n", i);
        print_output(activations + next_activations_offset, HIDDEN_DIM);
        activations_offset = next_activations_offset;
        params_offset += hidden_layer_param_count;
    }
    vec_mat_mul_softmax(
        params + params_offset, 
        activations + activations_offset, 
        out,
        HIDDEN_DIM,
        OUTPUT_DIM
    );
    printf("Output layer:\n");
    print_output(out, OUTPUT_DIM);
}

int main()
{
    float* params = init_params();
    float in[INPUT_DIM] = {1.0f};
    float out[OUTPUT_DIM] = {0.0f};
    float activations[HIDDEN_DIM * NUM_HIDDEN_LAYERS];
    forward_pass(params, in, out, activations);
    for(int i = 0; i < OUTPUT_DIM; i++)
    {
        printf("%f ", out[i]);
    }
    free(params);
}
