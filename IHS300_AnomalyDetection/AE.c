#include "AE.h"

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

// Encoder: Processes the input sequence
void encoder(LSTMCell* cells, int num_layers, float** inputs, int input_size, int hidden_size,
    int sequence_length, float** h_final, float** c_final) {
    float* h_prev = (float*)calloc(hidden_size, sizeof(float));
    float* c_prev = (float*)calloc(hidden_size, sizeof(float));
    float* h_next = (float*)calloc(hidden_size, sizeof(float));
    float* c_next = (float*)calloc(hidden_size, sizeof(float));

    for (int t = 0; t < sequence_length; ++t) {
        float* current_input = inputs[t];  // Input at time t

        for (int l = 0; l < num_layers; ++l) {
            LSTMCell* cell = &cells[l];

            // Pass the current input through the current layer's LSTM cell
            lstm_cell_forward(cell, current_input, h_prev, c_prev, h_next, c_next, input_size, hidden_size);

            // Prepare the input for the next layer in the stack (i.e., use the current layer's h_next)
            current_input = h_next;

            // Update h_prev and c_prev for the next layer
            for (int i = 0; i < hidden_size; ++i) {
                h_prev[i] = h_next[i];
                c_prev[i] = c_next[i];
            }
        }
    }

    // Store the final hidden and cell states for each layer
    for (int l = 0; l < num_layers; ++l) {
        for (int i = 0; i < hidden_size; ++i) {
            h_final[l][i] = h_next[i];
            c_final[l][i] = c_next[i];
        }
    }

    free(h_prev);
    free(c_prev);
    free(h_next);
    free(c_next);
}

// Decoder: Generates the output sequence
void decoder(LSTMCell* cells, int num_layers, float* encoded_vector, int input_size,
    int hidden_size, int sequence_length, float** outputs) {
    float* h_prev = (float*)calloc(hidden_size, sizeof(float));
    float* c_prev = (float*)calloc(hidden_size, sizeof(float));
    float* h_next = (float*)calloc(hidden_size, sizeof(float));
    float* c_next = (float*)calloc(hidden_size, sizeof(float));

    // Use the encoded vector as input for the decoder
    float* current_input = encoded_vector;

    for (int t = 0; t < sequence_length; ++t) {
        for (int l = 0; l < num_layers; ++l) {
            LSTMCell* cell = &cells[l];

            // Pass the encoded vector (or previous hidden state) through the current layer's LSTM cell
            lstm_cell_forward(cell, current_input, h_prev, c_prev, h_next, c_next, input_size, hidden_size);

            // Prepare the input for the next layer in the stack (i.e., use the current layer's h_next)
            current_input = h_next;

            // Update h_prev and c_prev for the next layer
            for (int i = 0; i < hidden_size; ++i) {
                h_prev[i] = h_next[i];
                c_prev[i] = c_next[i];
            }
        }

        // Store the output at time t
        for (int i = 0; i < hidden_size; ++i) {
            outputs[t][i] = h_next[i];
        }
    }

    free(h_prev);
    free(c_prev);
    free(h_next);
    free(c_next);
}
