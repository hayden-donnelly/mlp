#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "mnist.hpp"

#define INPUT_DIM 784 // 28*28
#define NUM_HIDDEN_LAYERS 3
#define HIDDEN_DIM 256
#define OUTPUT_DIM 10
#define BATCH_SIZE 1
#define LEARNING_RATE 0.000003f
#define NUM_EPOCHS 10

const int input_layer_weight_count = INPUT_DIM * HIDDEN_DIM;
const int hidden_layer_weight_count = HIDDEN_DIM * HIDDEN_DIM;
const int output_layer_weight_count = HIDDEN_DIM * OUTPUT_DIM;
 
const double two_pi = 2.0*3.14159265358979323846;

typedef struct
{
    float* weights;
    float* weight_grads;
    float* activations_out;
    float* activations;
    int num_activations;
    int num_weights;
} params_t;

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

params_t* init_params()
{
    params_t* params = (params_t*)malloc(sizeof(params_t));
    params->num_weights = 
        (NUM_HIDDEN_LAYERS * hidden_layer_weight_count)
        + input_layer_weight_count + output_layer_weight_count;
    
    const size_t weights_size = sizeof(float) * params->num_weights;
    params->weights = (float*)malloc(weights_size);
    params->weight_grads = (float*)malloc(weights_size);
    for(int i = 0; i < params->num_weights; i++)
    {
        params->weights[i] = random_normal();
        params->weight_grads[i] = 0.0f;
    }

    params->activations_out = (float*)malloc(sizeof(float) * OUTPUT_DIM);
    for(int i = 0; i < OUTPUT_DIM; i++)
    {
        params->activations_out[i] = 0.0f;
    }

    params->num_activations = (NUM_HIDDEN_LAYERS+1) * HIDDEN_DIM;
    params->activations = (float*)malloc(sizeof(float) * params->num_activations);
    for(int i = 0; i < params->num_activations; i++)
    {
        params->activations[i] = 0.0f;
    }
    return params;
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
        vec_in_grad[in_idx] = 0.0f;
        for(int out_idx = 0; out_idx < out_dim; out_idx++)
        {
            // df_j/dx_i = W_ij.
            // dL/dx_i = [df_1/dx_i * dL/df_1, df_2/dx_i * dL/df_2, ... , df_n/dx_i * dL/df_n.
            // Take a transposed view of the matrix.
            vec_in_grad[in_idx] += mat[out_idx * in_dim + in_idx] * vec_mat_mul_grad[out_idx];
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

void mse(float* label, float* pred_label, float* loss, const int num_classes)
{
    float error_sum = 0.0f;
    for(int i = 0; i < num_classes; i++)
    {
        const float error = label[i] - pred_label[i];
        error_sum += error * error;
    }
    *loss = error_sum / (float)num_classes; 
}

void mse_backward(float* label, float* pred_label, float* pred_label_grad, const int num_classes)
{
    for(int i = 0; i < num_classes; i++)
    {
        pred_label_grad[i] = 2.0f * (pred_label[i] - label[i]) / (float)num_classes;
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

void print_layer(float* weights, const int width, const int height)
{
    int weight_offset = 0;
    for(int i = 0; i < width; i++)
    {
        for(int k = 0; k < height; k++)
        {
            printf("%f ", weights[weight_offset++]);
        }
        printf("\n");
    }
}

void update_weights(
    float* weights, const float* weight_grads, 
    const float learning_rate, const int num_weights
){
    for(int i = 0; i < num_weights; i++)
    {
        weights[i] -= -learning_rate * weight_grads[i];
    }
}

void forward_pass(params_t* params, float* in)
{
    vec_mat_mul_sigmoid(params->weights, in, params->activations, INPUT_DIM, HIDDEN_DIM);
    int activations_offset = 0;
    int weight_offset = input_layer_weight_count;
    for(int i = 0; i < NUM_HIDDEN_LAYERS; i++)
    {
        const int next_activations_offset = activations_offset + HIDDEN_DIM;
        vec_mat_mul_sigmoid(
            params->weights + weight_offset, 
            params->activations + activations_offset, 
            params->activations + next_activations_offset,
            HIDDEN_DIM,
            HIDDEN_DIM
        );
        activations_offset = next_activations_offset;
        weight_offset += hidden_layer_weight_count;
    }
    vec_mat_mul_sigmoid(
        params->weights + weight_offset, 
        params->activations + activations_offset, 
        params->activations_out,
        HIDDEN_DIM,
        OUTPUT_DIM
    );
}

void backward_pass(params_t* params, float* vec_in, float* out_grad, const float learning_rate)
{
    float activation_grad[HIDDEN_DIM] = {0.0f};
    int activation_offset = params->num_activations - HIDDEN_DIM;
    int weight_offset = params->num_weights - output_layer_weight_count;
    vec_mat_mul_sigmoid_backward(
        params->weights + weight_offset, 
        params->activations + activation_offset,
        params->activations_out,
        out_grad,
        params->weight_grads + weight_offset,
        activation_grad,
        HIDDEN_DIM,
        OUTPUT_DIM
    );
    update_weights(
        params->weights + weight_offset,
        params->weight_grads + weight_offset,
        learning_rate,
        HIDDEN_DIM*OUTPUT_DIM
    );
    
    activation_offset -= HIDDEN_DIM;
    for(int i = 0; i < NUM_HIDDEN_LAYERS; i++)
    {
        weight_offset -= hidden_layer_weight_count;
        const int next_activation_offset = activation_offset - HIDDEN_DIM;
        vec_mat_mul_sigmoid_backward(
            params->weights + weight_offset, 
            params->activations + next_activation_offset,
            params->activations + activation_offset,
            activation_grad,
            params->weight_grads + weight_offset,
            activation_grad,
            HIDDEN_DIM,
            HIDDEN_DIM
        );
        update_weights(
            params->weights + weight_offset,
            params->weight_grads + weight_offset,
            learning_rate,
            HIDDEN_DIM*HIDDEN_DIM
        );
        activation_offset = next_activation_offset;
    }

    float in_grad[INPUT_DIM] = {0.0f};
    vec_mat_mul_sigmoid_backward(
        params->weights,
        vec_in,
        params->activations,
        activation_grad,
        params->weight_grads,
        in_grad,
        INPUT_DIM,
        HIDDEN_DIM
    );
    update_weights(
        params->weights,
        params->weight_grads,
        learning_rate,
        INPUT_DIM*HIDDEN_DIM
    );
}

int main()
{
    load_mnist();
    print_image(train_image[2]);
    params_t* params = init_params();
    
    for(int k = 0; k < NUM_EPOCHS; k++)
    {
        float accumulated_loss = 0.0f;
        for(int i = 0; i < MNIST_NUM_TRAIN; i++)
        {
            float out_grad[OUTPUT_DIM] = {0.0f};
            float loss = 0.0f;
            float* in = train_image[i];
            float label[OUTPUT_DIM] = {0.0f};
            label[train_label[i]] = 1.0f;

            forward_pass(params, in);
            mse(label, params->activations_out, &loss, OUTPUT_DIM);
            accumulated_loss += loss;
            //printf("MSE: %f\n", loss);
            mse_backward(label, params->activations_out, out_grad, OUTPUT_DIM);
            backward_pass(params, in, out_grad, LEARNING_RATE);
        }
        printf("Epoch %d, Loss %f", k, accumulated_loss/MNIST_NUM_TRAIN);
    }

    free(params->activations_out);
    free(params->activations);
    free(params->weights);
    free(params->weight_grads);
    free(params);
}
