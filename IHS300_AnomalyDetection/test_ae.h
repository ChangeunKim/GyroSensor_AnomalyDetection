
/*

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "read_csv.h"
#include "AE.h"
#include "eval.h"

void print_array(int* array, int size) {
    printf("1D Array:\n");
    for (int i = 0; i < size; i++) {
        printf("%d ", array[i]);
    }
}

void print_2d_array(float** array, int rows, int cols) {
    printf("2D Array (%d x %d):\n", rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", array[i][j]);
        }
        printf("\n");
    }
}

void print_evaluate(float* result) {
    printf("Accuracy: %.5f \n", result[ACCURACY]);
    printf("Precision: %.5f \n", result[PRECISION]);
    printf("Recall: %.5f \n", result[RECALL]);
    printf("F1 score: %.5f \n", result[F1_SCORE]);
}

void test_ae() {

}