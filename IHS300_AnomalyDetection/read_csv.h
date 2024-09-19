#ifndef READ_CSV_H
#define READ_CSV_H

#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Read constant value from csv file
double load_csv_constant(const char* filename);

// Read 1D array from csv file
void load_csv_1d(const char* filename, double** data, int* size);

// Read 2D array from csv file
void load_csv_2d(const char* filename, double*** data, int* rows, int* cols);

#endif