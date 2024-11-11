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

int64_t* make_random_labels(int n_labels)
{
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int64_t> dist(0, 9);
    
    int64_t* labels = (int64_t*)malloc(sizeof(int64_t) * n_labels);
    for(int i = 0; i < n_labels; ++i)
    {
        labels[i] = dist(rng); 
    }
    return labels;
}

bool matrices_are_equal(float* A, float* B, int n_elements, float eps)
{
    for(int i = 0; i < n_elements; ++i)
    {
        if(fabs(A[i] - B[i]) > eps)
        {
            return false;
        }
    }
    return true;
}
