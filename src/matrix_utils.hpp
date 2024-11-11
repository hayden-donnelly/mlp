#pragma once

float* make_random_matrix(int n_elements);
bool matrices_are_equal(float* A, float* B, int n_elements, float eps);
int64_t* make_random_labels(int n_labels);
