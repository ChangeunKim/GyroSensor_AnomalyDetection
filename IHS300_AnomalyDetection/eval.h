#ifndef EVAL_H
#define EVAL_H

/*
	eval.h

	Evaluate the model performance of multi-class classification and anomaly detection models.
*/

// Define macros for evaluation results
#define ACCURACY  0
#define PRECISION 1
#define RECALL	  2
#define F1_SCORE  3

// Evaluation function for anomaly detection tasks
void evaluate_anomaly(int* y_true, int* y_pred, int size, float* results);

#endif
