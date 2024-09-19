#ifndef SVM_H
#define SVM_H

#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>

#include "read_csv.h"

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
double compute_gamma(double** data, int num_samples, int num_features, int option);

// Function to compute the dot product
double dot_product(double* vec1, double* vec2, int size);
// Function to compute polynomial kernel
double polynomial_kernel(double* vec1, double* vec2, int degree, int size);
// Function to compute RBF kernel
double rbf_kernel(double* vec1, double* vec2, double gamma, int size);

// Function to predict the class for a new sample
void predict_multi_class(double** samples, int num_samples, double*** support_vectors, double** coefficients, double** intercepts, 
	int num_classes, int num_support_vectors, int num_features, int* predictions, int kernel);
void predict_one_class(double** samples, int num_samples, double** support_vectors, double intercept, 
	int num_support_vectors, int num_features, int* predictions, double nu, int kernel);

// SVM Function
void SVM(double** samples, int num_samples, int sample_features, int* predictions, int mode);

#endif