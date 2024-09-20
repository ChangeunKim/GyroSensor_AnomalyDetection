
/*

*/

#include "read_csv.h"
#include "SVM.h"
#include "eval.h"

void print_array(int* array, int size) {
    printf("1D Array:\n");
    for (int i = 0; i < size; i++) {
        printf("%d ", array[i]);
    }
}

void print_2d_array(double** array, int rows, int cols) {
    printf("2D Array (%d x %d):\n", rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", array[i][j]);
        }
        printf("\n");
    }
}

void print_evaluate(double* result) {
    printf("Accuracy: %.5f \n", result[ACCURACY]);
    printf("Precision: %.5f \n", result[PRECISION]);
    printf("Recall: %.5f \n", result[RECALL]);
    printf("F1 score: %.5f \n", result[F1_SCORE]);
}

int main() {
    // Load samples for prediction
    double** samples;
    int num_samples, sample_features;

    load_csv_2d("data/anomaly_input_sample.csv", &samples, &num_samples, &sample_features);
    // print_2d_array(samples, num_samples, sample_features);
    // printf("num_samples: %d \n", num_samples);
    // printf("sample_features: %d \n", sample_features);

	// Array to hold predictions
	int* predictions = (int*)malloc(num_samples * sizeof(int));

    // Predict using SVM
    SVM(samples, num_samples, sample_features, predictions, ONE_CLASS);

    // Load true labels for evaluation
    double* temp;
    int size;
    load_csv_1d("data/anomaly_target_sample.csv", &temp, &size);

    // Convert double array to integer array
    int* y_true = (int*)malloc((size) * sizeof(int));
    for (int i = 0; i < size; i++) {
        y_true[i] = (int)temp[i];  // Casting
    }
    // print_array(y_true, size);
    // printf("num_samples: %d \n", size);

    // See if number of samples in input and target matches
    if (num_samples != size) {
        printf("Exception: Size of input and target arrays doesn't match! \n");
        return 1;
    }

    // Evaluate prediction performance
    double results[4];
    evaluate_anomaly(y_true, predictions, size, results);

    print_evaluate(results);

	return 0;
}