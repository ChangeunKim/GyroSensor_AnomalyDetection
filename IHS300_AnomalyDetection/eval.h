#ifndef EVAL_H
#define EVAL_H

// Define macros for evaluation results
#define ACCURACY  0
#define PRECISION 1
#define RECALL	  2
#define F1_SCORE  3

// Evaluation function for classification and anomaly detection tasks
void evaluate(int* y_true, int* y_pred, int size, double* results);

#endif
