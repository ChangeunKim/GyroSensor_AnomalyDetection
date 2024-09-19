#include "eval.h"

// Evaluation function for classification and anomaly detection tasks
void evaluate(int* y_true, int* y_pred, int size, double* results){
    int true_positive = 0;
    int true_negative = 0;
    int false_positive = 0;
    int false_negative = 0;

    for (int i = 0; i < size; i++) {
        if (y_true[i] == 1 && y_pred[i] == 1) {
            true_positive++;
        }
        else if (y_true[i] == 0 && y_pred[i] == 0) {
            true_negative++;
        }
        else if (y_true[i] == 0 && y_pred[i] == 1) {
            false_positive++;
        }
        else if (y_true[i] == 1 && y_pred[i] == 0) {
            false_negative++;
        }
    }

    // Calculate metrics
    results[ACCURACY] = (double)(true_positive + true_negative) / size;
    results[PRECISION] = (true_positive + false_positive) ?
        (double)true_positive / (true_positive + false_positive) : 0.0;
    results[RECALL] = (true_positive + false_negative) ?
        (double)true_positive / (true_positive + false_negative) : 0.0;
    results[F1_SCORE] = (results[PRECISION] + results[RECALL]) ?
        2 * (results[PRECISION] * results[RECALL]) / (results[PRECISION] + results[RECALL]) : 0.0;

}
