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

static inline void vec_mat_mul(
    float* mat, float* vec_in, float* vec_out, 
    const int in_dim, const int out_dim
){
    int mat_offset = 0;
    for(int x = 0; x < in_dim; x++)
    {
        for(int y = 0; y < out_dim; y ++)
        {
            vec_out[x] += mat[mat_offset++] * vec_in[y];
        }
    }
}

static inline void relu(float* vec_in, float* vec_out, const int dim)
{
    for(int i = 0; i < dim; i++)
    {
        vec_out[i] = (vec_in[i] > 0.0f) ? vec_in[i] : 0.0f;
    }
}

void forward_pass(float* params, float* in, float* out, float* activations)
{
    vec_mat_mul(params, in, activations, INPUT_DIM, HIDDEN_DIM);
    int activations_offset = 0;
    int params_offset = input_layer_param_count;
    for(int i = 0; i < NUM_HIDDEN_LAYERS; i++)
    {
        const int next_activations_offset = activations_offset + HIDDEN_DIM;
        vec_mat_mul(
            &params[params_offset], 
            &activations[activations_offset], 
            &activations[next_activations_offset],
            HIDDEN_DIM,
            HIDDEN_DIM
        );
        activations_offset = next_activations_offset;
        params_offset += hidden_layer_param_count;
    }
    vec_mat_mul(
        &params[params_offset], 
        &activations[activations_offset], 
        out,
        HIDDEN_DIM,
        OUTPUT_DIM
    );
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
