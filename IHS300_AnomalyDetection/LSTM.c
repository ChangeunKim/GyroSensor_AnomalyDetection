#include "LSTM.h"

// Sigmoid function
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

// Hyperbolic tangent function (tanh)
float tanh_func(float x) {
    return tanh(x);
}

// LSTM cell forward pass for a single layer
void lstm_cell_forward(LSTMCell* cell, float* x, float* h_prev, float* c_prev,
    float* h_next, float* c_next, int input_size, int hidden_size) {
    // Temporary storage
    float* ft = (float*)malloc(hidden_size * sizeof(float));
    float* it = (float*)malloc(hidden_size * sizeof(float));
    float* ct = (float*)malloc(hidden_size * sizeof(float));
    float* ot = (float*)malloc(hidden_size * sizeof(float));

    float* temp_f = (float*)malloc(hidden_size * sizeof(float));
    float* temp_i = (float*)malloc(hidden_size * sizeof(float));
    float* temp_c = (float*)malloc(hidden_size * sizeof(float));
    float* temp_o = (float*)malloc(hidden_size * sizeof(float));

    // Forget gate: ft = sigmoid(Wf * x + Uf * h_prev + bf)
    mat_vec_mul(ft, cell->Wf, x, hidden_size, input_size);
    mat_vec_mul(temp_f, cell->Uf, h_prev, hidden_size, hidden_size);
    vector_add(ft, ft, temp_f, hidden_size);
    vector_add(ft, ft, cell->bf, hidden_size);
    for (int i = 0; i < hidden_size; ++i) {
        ft[i] = sigmoid(ft[i]);
    }

    // Input gate: it = sigmoid(Wi * x + Ui * h_prev + bi)
    mat_vec_mul(it, cell->Wi, x, hidden_size, input_size);
    mat_vec_mul(temp_i, cell->Ui, h_prev, hidden_size, hidden_size);
    vector_add(it, it, temp_i, hidden_size);
    vector_add(it, it, cell->bi, hidden_size);
    for (int i = 0; i < hidden_size; ++i) {
        it[i] = sigmoid(it[i]);
    }

    // Cell state candidate: ct = tanh(Wc * x + Uc * h_prev + bc)
    mat_vec_mul(ct, cell->Wc, x, hidden_size, input_size);
    mat_vec_mul(temp_c, cell->Uc, h_prev, hidden_size, hidden_size);
    vector_add(ct, ct, temp_c, hidden_size);
    vector_add(ct, ct, cell->bc, hidden_size);
    for (int i = 0; i < hidden_size; ++i) {
        ct[i] = tanh_func(ct[i]);
    }

    // Output gate: ot = sigmoid(Wo * x + Uo * h_prev + bo)
    mat_vec_mul(ot, cell->Wo, x, hidden_size, input_size);
    mat_vec_mul(temp_o, cell->Uo, h_prev, hidden_size, hidden_size);
    vector_add(ot, ot, temp_o, hidden_size);
    vector_add(ot, ot, cell->bo, hidden_size);
    for (int i = 0; i < hidden_size; ++i) {
        ot[i] = sigmoid(ot[i]);
    }

    // New cell state: c_next = ft * c_prev + it * ct
    for (int i = 0; i < hidden_size; ++i) {
        c_next[i] = ft[i] * c_prev[i] + it[i] * ct[i];
    }

    // New hidden state: h_next = ot * tanh(c_next)
    for (int i = 0; i < hidden_size; ++i) {
        h_next[i] = ot[i] * tanh_func(c_next[i]);
    }

    // Free temporary storage
    free(ft);
    free(it);
    free(ct);
    free(ot);
    free(temp_f);
    free(temp_i);
    free(temp_c);
    free(temp_o);
}

// Function to initialize the LSTM cell weights and biases
LSTMCell initialize_lstm_cell(int input_size, int hidden_size) {
    LSTMCell cell;
    cell.Wf = (float*)calloc(hidden_size * input_size, sizeof(float));
    cell.Wi = (float*)calloc(hidden_size * input_size, sizeof(float));
    cell.Wc = (float*)calloc(hidden_size * input_size, sizeof(float));
    cell.Wo = (float*)calloc(hidden_size * input_size, sizeof(float));
    cell.Uf = (float*)calloc(hidden_size * hidden_size, sizeof(float));
    cell.Ui = (float*)calloc(hidden_size * hidden_size, sizeof(float));
    cell.Uc = (float*)calloc(hidden_size * hidden_size, sizeof(float));
    cell.Uo = (float*)calloc(hidden_size * hidden_size, sizeof(float));
    cell.bf = (float*)calloc(hidden_size, sizeof(float));
    cell.bi = (float*)calloc(hidden_size, sizeof(float));
    cell.bc = (float*)calloc(hidden_size, sizeof(float));
    cell.bo = (float*)calloc(hidden_size, sizeof(float));

    // Initialize weights and biases (e.g., random initialization)

    return cell;
}
