#ifndef AE_H
#define AE_H

/*
	AE.h

	C implementation of autoencoder encoding and decoding logic given model files.
*/

#include "read_csv.h"
#include "LSTM.h"

// Function to initialize the LSTM cell weights and biases
LSTMCell initialize_lstm_cell(int input_size, int hidden_size);
// Encoder: Processes the input sequence
void encoder(LSTMCell* cells, int num_layers, float** inputs, int input_size, int hidden_size,
	int sequence_length, float** h_final, float** c_final);
// Decoder: Generates the output sequence
void decoder(LSTMCell* cells, int num_layers, float* encoded_vector, int input_size,
	int hidden_size, int sequence_length, float** outputs);

#endif