#include <stdio.h>

#define NUM_HIDDEN_LAYERS 3
#define HIDDEN_DIM 3
#define OUTPUT_DIM 10

static inline void hidden_layer_forward(float* params_this_layer, float* in, float* out)
{
    int offset = 0;
    for(int x = 0; x < HIDDEN_DIM; x++)
    {
        for(int y = 0; y < HIDDEN_DIM; y ++)
        {
            out[x] += params_this_layer[offset++] * in[y];
        }
    }
}

int main()
{
    printf("Hello World\n");
    float params[9] = 
    {
        1.0f, 0.0f, 0.0f,
        0.0f, 2.0f, 0.0f,
        0.0f, 0.0f, 1.0f
    };
    float in[3] = {1.0f, 2.0f, 3.0f};
    float out[3] = {0.0f};
    hidden_layer_forward(params, in, out);
    for(int i = 0; i < HIDDEN_DIM; i++)
    {
        printf("%f ", out[i]);
    }
    printf("\n");
}
