#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>

#define N_COEFF 11

#include "components.h"

void fir(const int* signal_r, // Read-Only Vector 1
           int* output_r,       // Output Result
           int signal_length,   // Number of elements
           int mul1,
           int add1
) {
    //int coeff_reg[N_COEFF];
    int coeff_reg[N_COEFF] = {53, 0, -91, 0, 313, 500, 313, 0, -91, 0, 53};
    int x1;

    int shift_reg[N_COEFF];

    init_loop:
    for (int i = 0; i < N_COEFF; i++) {
        shift_reg[i] = 0;
        //coeff_reg[i] = coeff[i];
    }

    outer_loop:
    for (int j = 0; j < signal_length; j++) {
        int acc = 0;
        int x = signal_r[j];

        shift_loop:
        for (int i = N_COEFF - 1; i >= 0; i--) {
            if (i == 0) {
                x1 = (int)mul[mul1]((uint16_t)x, (uint16_t)coeff_reg[0]);
                acc = (int)add[add1]((uint16_t)acc, (uint16_t)x1);
                shift_reg[0] = x;
            } else {
                shift_reg[i] = shift_reg[i - 1];
                x1 = (int)mul[mul1]((uint16_t)shift_reg[i], (uint16_t)coeff_reg[i]);
                acc = (int)add[add1]((uint16_t)acc, (uint16_t)x1);
            }
        }
        output_r[j] = acc;
    }
}

void initialize(int* in1, // Read-Only Vector 1
                int size,   // Number of elements
                int seed
) {

    srand(seed);
    int val = 256;
    for (int i = 0; i < size; ++i) {
        in1[i] = rand() % val;
    }
}

// Output to stdout
void print(int *r, unsigned size) {
    for (int i = 0; i < size; i++) {
        printf(" %d", r[i]);
    }
    //printf("\n");
}

int main(int argc, char *argv[]) {
    FILE *input, *output;
    int *signal, *out;
    unsigned seed, size;

    if ((input = fopen(argv[1], "r")) == NULL) {
        fprintf(stderr, "Error: could not open input file\n");
        return 1;
    }


    fscanf(input, "%u", &size);
    fscanf(input, "%u", &seed);

    // Allocate matrices
    signal = (int *) malloc(sizeof(int) * size);
    out = (int *) malloc(sizeof(int) * size);

    initialize(signal, size, seed);

    fir(signal, out, size, atoi(argv[2]), atoi(argv[3]));

    // Write the final image to disk
    print(out, size);

    free(signal);

    return EXIT_SUCCESS;
}
