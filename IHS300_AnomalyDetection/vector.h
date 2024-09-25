#ifndef VECTOR_H
#define VECTOR_H

void mat_vec_mul(float* result, float* matrix, float* vector, int rows, int cols);
void vector_add(float* result, float* vec1, float* vec2, int size);
// Function to compute the dot product
float dot_product(float* vec1, float* vec2, int size);

#endif

