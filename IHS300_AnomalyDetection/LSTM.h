#ifndef LSTM_H
#define LSTM_H

#include <math.h>
#include "vector.h"

// Structure to hold LSTM parameters
typedef struct {
    float* Wf, * Wi, * Wc, * Wo;  // Weights for forget, input, cell state, output gates
    float* Uf, * Ui, * Uc, * Uo;  // Recurrent weights
    float* bf, * bi, * bc, * bo;  // Biases for forget, input, cell state, output gates
} LSTMCell;


float sigmoid(float x);
float tanh_func(float x);
void lstm_cell_forward(LSTMCell* cell, float* x, float* h_prev, float* c_prev,
    float* h_next, float* c_next, int input_size, int hidden_size);


#endif