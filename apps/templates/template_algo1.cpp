#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "../../apps/templates/mul16s_HFZ.h"
#include "../../apps/templates/mul16s_GK2.h"
#include "../../apps/templates/mul16s_GAT.h"
#include "../../apps/templates/mul16s_HDG.h"
#include "../../apps/templates/mul16s_HHP.h"
#include "../../apps/templates/mul16s_G80.h"
#include "../../apps/templates/mul16s_G7F.h"
#include "../../apps/templates/mul16s_G7Z.h"
#include "../../apps/templates/mul16s_HEB.h"

#include "../../apps/templates/add16se_2GE.h"
#include "../../apps/templates/add16se_2KV.h"
#include "../../apps/templates/add16se_2DN.h"
#include "../../apps/templates/add16se_25S.h"
#include "../../apps/templates/add16se_2AS.h"
#include "../../apps/templates/add16se_2JB.h"
#include "../../apps/templates/add16se_294.h"
#include "../../apps/templates/add16se_2JY.h"
#include "../../apps/templates/add16se_20J.h"
#include "../../apps/templates/add16se_1Y7.h"
#include "../../apps/templates/add16se_259.h"
#include "../../apps/templates/add16se_26Q.h"
#include "../../apps/templates/add16se_29A.h"
#include "../../apps/templates/add16se_2E1.h"
#include "../../apps/templates/add16se_28H.h"
#include "../../apps/templates/add16se_2BY.h"
#include "../../apps/templates/add16se_2LJ.h"
#include "../../apps/templates/add16se_2H0.h"
#include "../../apps/templates/add16se_RCA.h"

#include "ap_int.h"

extern "C" {
void algo1(const ap_int<16>* in1, // Read-Only Vector 1
				const ap_int<16>* in2, // Read-Only Vector 2
			    const ap_int<16>* in3, // Read-Only Vector 3
			    const ap_int<16>* in4, // Read-Only Vector 4
			    const ap_int<16>* in5, // Read-Only Vector 5
				ap_int<16>* out,       // Output Result
                int elements    // Number of elements
               ) {
#pragma HLS INTERFACE mode=m_axi bundle=gmem0 port=in1 depth=4096
#pragma HLS INTERFACE mode=m_axi bundle=gmem1 port=in2 depth=4096
#pragma HLS INTERFACE mode=m_axi bundle=gmem2 port=in3 depth=4096
#pragma HLS INTERFACE mode=m_axi bundle=gmem3 port=in4 depth=4096
#pragma HLS INTERFACE mode=m_axi bundle=gmem4 port=in5 depth=4096
#pragma HLS INTERFACE mode=m_axi bundle=gmem0 port=out depth=4096

	ap_int<16> x5, x6, x7;

dfg1:
    for (int i = 0; i < (elements/8)*8; i++) {
#pragma HLS LOOP_TRIPCOUNT avg=4096 max=4096 min=4096
#pragma HLS UNROLL factor=8
    	x5 = mul16s_"mul1"(in1[i], in2[i]);
    	x6 = add16se_"add1"(in3[i], in4[i]);
    	if (x6 == 0)
    		x7 = 0;
    	else
    		x7 = x5 / x6;
        out[i] = mul16s_"mul2"(in5[i], x7);
    }
}
}
