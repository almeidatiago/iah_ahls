#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <assert.h>
//#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>

#include "components.h"

void algo1(const int* in1, // Read-Only Vector 1
           const int* in2, // Read-Only Vector 2
           const int* in3, // Read-Only Vector 3
           const int* in4, // Read-Only Vector 4
           const int* in5, // Read-Only Vector 5
           int* out,       // Output Result
           int elements,   // Number of elements
           int mul1, int mul2,
           int add1
          ) {
    
	int x5, x6, x7;
dfg1:
    for (int i = 0; i < elements; i++) {
    	x5 = (int)mul[mul1]((uint16_t)in1[i], (uint16_t)in2[i]);
    	x6 = (int)add[add1]((uint16_t)in3[i], (uint16_t)in4[i]);
    	if (x6 == 0)
    		x7 = 0;
    	else
    		x7 = x5 / x6;
        out[i] = (int)mul[mul2]((uint16_t)in5[i], (uint16_t)x7);
    }
}


void initialize(int* in1, // Read-Only Vector 1
           int* in2, // Read-Only Vector 2
           int* in3, // Read-Only Vector 3
           int* in4, // Read-Only Vector 4
           int* in5, // Read-Only Vector 5
           int size,   // Number of elements
           int seed
          ) {
    
    auto const s = seed;
    std::mt19937 gen{s};

    // Values near the mean are the most likely. Standard deviation
    // affects the dispersion of generated values from the mean.
    std::normal_distribution d{0.0, 30.0};
    
    //std::uniform_int_distribution<int>  d(-128, 127);

    // Draw a sample from the normal distribution and round it to an integer.
    auto random_int = [&d, &gen]{ return std::lround(d(gen)); };
    
  //srand(seed);
  //int val = 181;
  for (int i = 0; i < size; ++i) {
        in1[i] = random_int();
        in2[i] = random_int();
        in3[i] = random_int();
        in4[i] = random_int();
        in5[i] = random_int();
        /*in1[i] = rand() % val;
        in2[i] = rand() % val;
        in3[i] = rand() % val;
        in4[i] = rand() % val;
        in5[i] = rand() % val;*/
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
    FILE *input;
    int *in1, *in2, *in3, *in4, *in5, *out;
    unsigned seed, size;
    
    if ((input = fopen(argv[1], "r")) == NULL) {
        fprintf(stderr, "Error: could not open file\n");
        return 1;
    }
    
    // Read inputs
    fscanf(input, "%u", &size);
    fscanf(input, "%u", &seed);
    
    // Allocate matrices
    in1 = (int *)malloc(sizeof(int) * size);
    in2 = (int *)malloc(sizeof(int) * size);
    in3 = (int *)malloc(sizeof(int) * size);
    in4 = (int *)malloc(sizeof(int) * size);
    in5 = (int *)malloc(sizeof(int) * size);
    out = (int *)malloc(sizeof(int) * size);

    initialize(in1, in2, in3, in4, in5, size, seed);
    
    algo1(in1, in2, in3, in4, in5, out, size, atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
    
    print(out, size);
 
    return EXIT_SUCCESS;
}
