#ifndef SVM_H
#define SVM_H

/*
	SVM.h

	C implementation of SVM prediction logic given model files.
*/

#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>

#include "read_csv.h"
#include "vector.h"

// Define macros for SVM type
#define ONE_CLASS   1
#define MULTI_CLASS 2

// Define macros for kernel type
#define LINEAR  1
#define POLY    2
#define RBF     3

// Define macros for gamma computation options
#define AUTO  0
#define SCALE 1

// Function to compute gamma for RBF kernel
float compute_gamma(float** data, int num_samples, int num_features, int option);

// Function to compute polynomial kernel
float polynomial_kernel(float* vec1, float* vec2, int degree, int size);
// Function to compute RBF kernel
float rbf_kernel(float* vec1, float* vec2, float gamma, int size);

// Function to predict the class for a new sample
void predict_multi_class(float** samples, int num_samples, float*** support_vectors, float** coefficients, float** intercepts, 
	int num_classes, int num_support_vectors, int num_features, int* predictions, int kernel);
void predict_one_class(float** samples, int num_samples, float** support_vectors, float intercept, 
	int num_support_vectors, int num_features, int* predictions, float nu, int kernel);

// SVM Function
void SVM(float** samples, int num_samples, int sample_features, int* predictions, int mode);

#endif