#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "components.h"

void algo2(const int* in1, // Read-Only Vector 1
           const int* in2, // Read-Only Vector 2
           const int* in3, // Read-Only Vector 3
           const int* in4, // Read-Only Vector 4
           int* out,       // Output Result
           int elements,   // Number of elements
           int mul1, 
           int add1, int add2
          ) {
    
	int x5, x6;
dfg1:
    for (int i = 0; i < elements; i++) {        
        x5 = (int)mul[mul1]((uint16_t)in1[i], (uint16_t)in2[i]);
    	x6 = (int)add[add1]((uint16_t)in3[i], (uint16_t)in4[i]);
        out[i] = (int)add[add2]((uint16_t)x5, (uint16_t)x6);
    }
}

void initialize(int* in1, // Read-Only Vector 1
           int* in2, // Read-Only Vector 2
           int* in3, // Read-Only Vector 3
           int* in4, // Read-Only Vector 4
           int size,   // Number of elements
           int seed
          ) {
    
  srand(seed);
  int val = 181;
  for (int i = 0; i < size; ++i) {
        in1[i] = rand() % val;
        in2[i] = rand() % val;
        in3[i] = rand() % val;
        in4[i] = rand() % val;
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
    int *in1, *in2, *in3, *in4, *out;
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
    out = (int *)malloc(sizeof(int) * size);

    initialize(in1, in2, in3, in4, size, seed);
    
    algo2(in1, in2, in3, in4, out, size, atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
    
    print(out, size);
 
    return EXIT_SUCCESS;
}
