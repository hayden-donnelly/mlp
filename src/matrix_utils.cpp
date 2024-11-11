#include <random>
#include <cmath>

float* make_random_matrix(int n_elements)
{
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    float* mat = (float*)malloc(sizeof(float) * n_elements);
    for(int i = 0; i < n_elements; ++i)
    {
        mat[i] = dist(rng); 
    }
    return mat;
}

bool matrices_are_equal(float* A, float* B, int n_elements, float eps)
{
    for(int i = 0; i < n_elements; ++i)
    {
        if(fabs(A[i] - B[i]) > eps)
        {
            printf("failed at %d with A[i] = %f and B[i] = %f\n", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}
