#include "vector.h"

// Function to perform matrix-vector multiplication
void mat_vec_mul(float* result, float* matrix, float* vector, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        result[i] = 0.0;
        for (int j = 0; j < cols; ++j) {
            result[i] += matrix[i * cols + j] * vector[j];
        }
    }
}

// Function to add two vectors
void vector_add(float* result, float* vec1, float* vec2, int size) {
    for (int i = 0; i < size; ++i) {
        result[i] = vec1[i] + vec2[i];
    }
}

// Function to compute the dot product
float dot_product(float* vec1, float* vec2, int size) {
    float result = 0.0;
    for (int i = 0; i < size; i++) {
        result += vec1[i] * vec2[i];
    }
    return result;
}