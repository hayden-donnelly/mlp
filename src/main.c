#include <stdio.h>
#include "mnist.h"

#define NUM_HIDDEN_LAYERS 3
#define HIDDEN_DIM 3
#define OUTPUT_DIM 10
#define BATCH_SIZE 1

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

int main()
{
    printf("Hello World\n");
    float params[9] = 
    {
        1.0f, 0.0f, 0.0f,
        0.0f, -2.0f, 0.0f,
        0.0f, 0.0f, 1.0f
    };
    float in[3] = {1.0f, 2.0f, 3.0f};
    float out[3] = {0.0f};
    vec_mat_mul(params, in, out, HIDDEN_DIM, HIDDEN_DIM);
    relu(out, out, HIDDEN_DIM);
    for(int i = 0; i < HIDDEN_DIM; i++)
    {
        printf("%f ", out[i]);
    }
    printf("\n");
    load_mnist();
    for(int i = 0; i < NUM_TEST; i++)
    {
        printf("%d\n", test_label[i]);
    }
}
