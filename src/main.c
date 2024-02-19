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

static inline void vec_mat_mul_relu_norm(
    float* mat, float* vec_in, float* vec_out, 
    const int in_dim, const int out_dim
){
    int mat_offset = 0;
    float layer_sum = 0.0f;
    for(int out_idx = 0; out_idx < out_dim; out_idx++)
    {
        vec_out[out_idx] = 0.0f;
        for(int in_idx = 0; in_idx < in_dim; in_idx++)
        {
            vec_out[out_idx] += mat[mat_offset++] * vec_in[in_idx];
        }
        scalar_relu(vec_out + out_idx);
        layer_sum += vec_out[out_idx];
    }
    const float layer_mean = layer_sum / (float)out_dim;
    layer_sum = 0.0f;
    for(int i = 0; i < out_dim; i++)
    {
        const float mean_shifted = vec_out[i] - layer_mean;
        layer_sum += mean_shifted * mean_shifted;
    }
    const float sqrt_layer_variance = (float)sqrt((double)(layer_sum / (float)out_dim) + 0.00001);
    printf("var %f, mean %f\n", sqrt_layer_variance, layer_mean);
    for(int i = 0; i < out_dim; i++)
    {
        vec_out[i] = (vec_out[i] - layer_mean) / sqrt_layer_variance;
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
    vec_mat_mul_relu_norm(params, in, activations, INPUT_DIM, HIDDEN_DIM);
    printf("Input layer:\n");
    print_output(activations, HIDDEN_DIM);
    int activations_offset = 0;
    int params_offset = input_layer_param_count;
    for(int i = 0; i < NUM_HIDDEN_LAYERS; i++)
    {
        const int next_activations_offset = activations_offset + HIDDEN_DIM;
        vec_mat_mul_relu_norm(
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
