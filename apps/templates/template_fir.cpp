/**
* Copyright (C) 2019-2021 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

/*
  Finite Impulse Response(FIR) Filter

  This example demonstrates how to perform a shift register operation to
  implement a Finite Impulse Response(FIR) filter.

  Shift register operation is the cascading of values of an array by one or more
  places. Here is an example of what a shift register operation looks like on an
  array of length four:

                     ___       ___      ___      ___
  1. shift_reg[N] : | A |  <- | B | <- | C | <- | D |
                     ---       ---      ---      ---
                     ___       ___      ___      ___
  2. shift_reg[N] : | B |  <- | C | <- | D | <- | D |
                     ---       ---      ---      ---

  Here each of the values are copied into the register on the left. This type of
  operation is useful when you want to work on a sliding window of data or when
  the data is being streamed into the kernel.

  The Xilinx compiler can recognize this type of operation into the appropriate
  hardware. For example, the previous illustration can be coded using the
  following loop:

  #define N 4

  __attribute__((opencl_unroll_hint))
  for(int i = 0; i < N-1; i++) {
      shift_reg[i] = shift_reg[i+1];
  }

  The compiler needs to know the number of registers at compile time so the
  definition of N must be a compile time variable.

*/
#include "../../src/mul16s_HFZ.h"
#include "../../src/mul16s_GK2.h"
#include "../../src/mul16s_GAT.h"
#include "../../src/mul16s_HDG.h"
#include "../../src/mul16s_HHP.h"
#include "../../src/mul16s_G80.h"
#include "../../src/mul16s_G7F.h"
#include "../../src/mul16s_G7Z.h"
#include "../../src/mul16s_HEB.h"

#include "../../src/add16se_2GE.h"
#include "../../src/add16se_2KV.h"
#include "../../src/add16se_2DN.h"
#include "../../src/add16se_25S.h"
#include "../../src/add16se_2AS.h"
#include "../../src/add16se_2JB.h"
#include "../../src/add16se_294.h"
#include "../../src/add16se_2JY.h"
#include "../../src/add16se_20J.h"
#include "../../src/add16se_1Y7.h"
#include "../../src/add16se_259.h"
#include "../../src/add16se_26Q.h"
#include "../../src/add16se_29A.h"
#include "../../src/add16se_2E1.h"
#include "../../src/add16se_28H.h"
#include "../../src/add16se_2BY.h"
#include "../../src/add16se_2LJ.h"
#include "../../src/add16se_2H0.h"
#include "../../src/add16se_RCA.h"

#include "ap_int.h"

// Number of coefficient components
#define N_COEFF 11
// FIR using shift register
extern "C" {
void fir(ap_int<16>* output_r, ap_int<16>* signal_r, int signal_length) {
    ap_int<16> coeff_reg[N_COEFF];
    ap_int<16> coeff[N_COEFF] = {53, 0, -91, 0, 313, 500, 313, 0, -91, 0, 53};
    ap_int<16> x1;

    // Partitioning of this array is required because the shift register
    // operation will need access to each of the values of the array in
    // the same clock. Without partitioning the operation will need to
    // be performed over multiple cycles because of the limited memory
    // ports available to the array.
    ap_int<16> shift_reg[N_COEFF];
#pragma HLS ARRAY_PARTITION variable = shift_reg complete dim = 0

init_loop:
    for (int i = 0; i < N_COEFF; i++) {
#pragma HLS PIPELINE II = 1
        shift_reg[i] = 0;
        coeff_reg[i] = coeff[i];
    }

outer_loop:
    for (int j = 0; j < signal_length; j++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
#pragma HLS PIPELINE II = 1
        int acc = 0;
        int x = signal_r[j];

    // This is the shift register operation. The N_COEFF variable is defined
    // at compile time so the compiler knows the number of operations
    // performed by the loop. This loop does not require the unroll
    // attribute because the outer loop will be automatically pipelined so
    // the compiler will unroll this loop in the process.
    shift_loop:
        for (int i = N_COEFF - 1; i >= 0; i--) {
            if (i == 0) {
                x1 = mul16s_"mul1"(x, coeff_reg[0]);
                acc = add16se_"add1"(acc, x1);
                shift_reg[0] = x;
            } else {
                shift_reg[i] = shift_reg[i - 1];
                x1 = mul16s_"mul1"(shift_reg[i], coeff_reg[i]);
                acc = add16se_"add1"(acc, x1);
            }
        }
        output_r[j] = acc;
    }
}
}
