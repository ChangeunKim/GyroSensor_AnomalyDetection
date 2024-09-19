#include "SVM.h"

// Function to compute gamma for RBF kernel
double compute_gamma(double** samples, int num_samples, int num_features, int option) {
    if (num_features <= 0) {
        fprintf(stderr, "Number of features must be greater than zero.\n");
        return -1.0; // Error case
    }

    if (option == AUTO) {
        return 1.0 / num_features; // Gamma for "auto"
    }
    else if (option == SCALE) {
        // Compute the standard deviation
        double mean = 0.0;
        for (int i = 0; i < num_samples; i++) {
            for (int j = 0; j < num_features; j++) {
                mean += samples[i][j];
            }
        }
        mean /= (num_samples * num_features);

        double variance = 0.0;
        for (int i = 0; i < num_samples; i++) {
            for (int j = 0; j < num_features; j++) {
                double diff = samples[i][j] - mean;
                variance += diff * diff;
            }
        }
        variance /= (num_samples * num_features);
        double std_dev = sqrt(variance);

        return 1.0 / (num_features * std_dev); // Gamma for "scale"
    }
    else {
        fprintf(stderr, "Invalid option for gamma. Use 'auto' or 'scale'.\n");
        return -1.0; // Error case
    }
}

// Function to compute the dot product
double dot_product(double* vec1, double* vec2, int size) {
    double result = 0.0;
    for (int i = 0; i < size; i++) {
        result += vec1[i] * vec2[i];
    }
    return result;
}

// Function to compute polynomial kernel
double polynomial_kernel(double* vec1, double* vec2, int degree, int size) {
    double dotp = dot_product(vec1, vec2, size);

    return pow(dotp + 1, degree);
}

// Function to compute RBF kernel
double rbf_kernel(double* vec1, double* vec2, double gamma, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        double diff = vec1[i] - vec2[i];
        sum += diff * diff;
    }
    return exp(-gamma * sum); // exp(-gamma * ||x1 - x2||^2)
}

// Function to predict the class for a new sample
void predict_multi_class(double** samples, int num_samples, double*** support_vectors, double** coefficients, double** intercepts, 
    int num_classes, int num_support_vectors, int num_features, int* predictions, int kernel) {

    // Hyperparameter for poly kernel and RBF kernel
    int degree = 4;
    double gamma = compute_gamma(samples, num_samples, num_features, SCALE);

    for (int s = 0; s < num_samples; s++) {
        double max_score = -DBL_MAX;
        int predicted_class = -1;

        // Iterate over each class
        for (int i = 0; i < num_classes; i++) {
            double score = 0.0;

            // Compute the score for the class
            for (int j = 0; j < num_support_vectors; j++) {
                switch (kernel) {
                case LINEAR:
                    score += coefficients[i][j] * dot_product(samples[s], support_vectors[i][j], num_features);
                    break;
                case POLY:
                    score += coefficients[i][j] * polynomial_kernel(samples[s], support_vectors[i][j], degree, num_features);
                    break;
                case RBF:
                    score += coefficients[i][j] * rbf_kernel(samples[s], support_vectors[i][j], gamma, num_features);
                    break;
                default:
                    score += coefficients[i][j] * dot_product(samples[s], support_vectors[i][j], num_features);
                    break;
                }
            }
            score += intercepts[i][0];  // Adjust with intercept

            // Check if this is the best score so far
            if (score > max_score) {
                max_score = score;
                predicted_class = i;
            }
        }

        predictions[s] = predicted_class;  // Store the class with the highest score
    }
}

// Prediction function implementation
void predict_one_class(double** samples, int num_samples, double** support_vectors, double intercept, 
    int num_support_vectors, int num_features, int* predictions, double nu, int kernel) {

    // Hyperparameter for poly kernel and RBF kernel
    int degree = 4;
    double gamma = compute_gamma(samples, num_samples, num_features, SCALE);

    for (int s = 0; s < num_samples; s++) {
        double sum = 0.0;

        // Compute the decision function for each sample
        for (int i = 0; i < num_support_vectors; i++) {
            switch (kernel) {
                case LINEAR:
                    sum += dot_product(samples[s], support_vectors[i], num_features);
                    break;
                case POLY:
                    sum += polynomial_kernel(samples[s], support_vectors[i], degree, num_features);
                    break;
                case RBF:
                    sum += rbf_kernel(samples[s], support_vectors[i], gamma, num_features);
                    break;
                default:
                    sum += dot_product(samples[s], support_vectors[i], num_features);
                    break;
                }
        }

        // Calculate threshold based on nu
        double threshold = nu * num_support_vectors + intercept;

        // Store prediction result
        predictions[s] = (sum >= threshold) ? 1 : -1;  // 1 for normal, -1 for outlier
    }
}


void SVM(double** samples, int num_samples, int sample_features, int* predictions, int mode) {

    // Dimension of input features
    int num_support_vectors, num_features;
    // Kernel type
    int kernel = RBF;

    if (mode == MULTI_CLASS) {
        // Number of classes in multi-class classification task
        int num_classes = 3;

        // Model parameters
        double** support_vectors = (double**)malloc(num_classes * sizeof(double*));
        double** coefficients = (double**)malloc(num_classes * sizeof(double*));
        double** intercepts = (double**)malloc(num_classes * sizeof(double*));

        // Load parameters for each class
        for (int i = 0; i < num_classes; i++) {
            load_csv_2d("support_vectors_class_{i}.csv", &support_vectors[i], &num_support_vectors, &num_features);
            load_csv_2d("coefficients_class_{i}.csv", &coefficients[i], &num_support_vectors, NULL);
            load_csv_2d("intercept_class_{i}.csv", &intercepts[i], NULL, NULL);
        }

        // Make a prediction
        predict_multi_class(samples, num_samples, support_vectors, coefficients, intercepts, 
            num_classes, num_support_vectors, num_features, predictions, kernel);

        // Free allocated memory
        for (int i = 0; i < num_classes; i++) {
            free(support_vectors[i]);
            free(coefficients[i]);
            free(intercepts[i]);
        }
        free(support_vectors);
        free(coefficients);
        free(intercepts);
    }
    else if (mode == ONE_CLASS){
        // Set nu value
        double nu = 0.01;

        // Load support vectors
        double* support_vectors;
        load_csv_2d("model/support_vectors.csv", &support_vectors, &num_support_vectors, &num_features);

        // Load intercept
        double intercept = load_csv_constant("model/intercept.csv");

        // Make a prediction
        predict_one_class(samples, num_samples, support_vectors, intercept,
            num_support_vectors, num_features, predictions, nu, kernel);

        // Free allocated memory
        free(support_vectors);
    }

    for (int i = 0; i < num_samples; i++) {
        free(samples[i]);
    }
    free(samples);

    return predictions;
}