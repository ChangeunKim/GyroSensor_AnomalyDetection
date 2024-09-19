#include "read_csv.h"

// Read constant value from csv file
double load_csv_constant(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Unable to open file");
        exit(EXIT_FAILURE);
    }

    double value;
    // Read the first (and only) value from the CSV file
    if (fscanf(file, "%lf", &value) != 1) {
        perror("Error reading value");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    fclose(file);
    return value;
}

// Read 1D array from csv file
void load_csv_1d(const char* filename, double** data, int* size) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Unable to open file");
        exit(EXIT_FAILURE);
    }

    // Initialize size
    *size = 0;

    // Count lines in the file to determine size
    while (!feof(file)) {
        if (fgetc(file) == '\n') {
            (*size)++;
        }
    }
    rewind(file); // Reset file pointer to the beginning

    // Allocate memory for the array
    *data = (double*)malloc((*size) * sizeof(double));

    // Read the data
    for (int i = 0; i < *size; i++) {
        fscanf(file, "%lf", &((*data)[i]));
    }

    fclose(file);
}

// Read 2D array from csv file
void load_csv_2d(const char* filename, double*** data, int* rows, int* cols) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Unable to open file");
        exit(EXIT_FAILURE);
    }

    // Initialize rows and cols
    *rows = 0;
    *cols = 0;

    // First pass to count rows and columns
    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), file)) {
        if (*rows == 0) {
            char* token = strtok(buffer, ",");
            while (token) {
                (*cols)++;
                token = strtok(NULL, ",");
            }
        }
        (*rows)++;
    }
    rewind(file); // Reset file pointer to the beginning

    // Allocate memory for the 2D array
    *data = (double**)malloc((*rows) * sizeof(double*));
    for (int i = 0; i < *rows; i++) {
        (*data)[i] = (double*)malloc((*cols) * sizeof(double));
    }

    // Read the data into the 2D array
    for (int i = 0; i < *rows; i++) {
        for (int j = 0; j < *cols; j++) {
            fscanf(file, "%lf", &((*data)[i][j]));
            if (j < *cols - 1) {
                fgetc(file); // Skip comma
            }
        }
    }

    fclose(file);
}