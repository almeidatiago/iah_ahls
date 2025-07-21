#include <stdint.h>
#include <stdlib.h>
#include "ap_int.h"

typedef ap_int<16> in16_t;
typedef ap_uint<1> in1_t;

in16_t add16se_RCA (in16_t  A, in16_t  B) {
    #pragma HLS inline off    
    
    in1_t dout0, dout1, dout2, dout3, dout4, dout5, dout6, dout7, dout8, dout9, dout10, dout11, dout12, dout13, dout14, dout15;
    in1_t cout0, cout1, cout2, cout3, cout4, cout5, cout6, cout7, cout8, cout9, cout10, cout11, cout12, cout13, cout14;//, cout15;
    in1_t xout1, xout2, xout3, xout4, xout5, xout6, xout7, xout8, xout9, xout10, xout11, xout12, xout13, xout14, xout15;
    in16_t O;

    dout0 = ((A >> 0)&1)^((B >> 0)&1);
    cout0 = ((A >> 0)&1)&((B >> 0)&1);
    xout1 = ((A >> 1)&1)^((B >> 1)&1);
    dout1 = xout1 ^ cout0;
    cout1 = (((A >> 1)&1)&((B >> 1)&1))|(xout1 & cout0);
    xout2 = ((A >> 2)&1)^((B >> 2)&1);
    dout2 = xout2 ^ cout1;
    cout2 = (((A >> 2)&1)&((B >> 2)&1))|(xout2 & cout1);
    xout3 = ((A >> 3)&1)^((B >> 3)&1);
    dout3 = xout3 ^ cout2;
    cout3 = (((A >> 3)&1)&((B >> 3)&1))|(xout3 & cout2);
    xout4 = ((A >> 4)&1)^((B >> 4)&1);
    dout4 = xout4 ^ cout3;
    cout4 = (((A >> 4)&1)&((B >> 4)&1))|(xout4 & cout3);
    xout5 = ((A >> 5)&1)^((B >> 5)&1);
    dout5 = xout5 ^ cout4;
    cout5 = (((A >> 5)&1)&((B >> 5)&1))|(xout5 & cout4);
    xout6 = ((A >> 6)&1)^((B >> 6)&1);
    dout6 = xout6 ^ cout5;
    cout6 = (((A >> 6)&1)&((B >> 6)&1))|(xout6 & cout5);
    xout7 = ((A >> 7)&1)^((B >> 7)&1);
    dout7 = xout7 ^ cout6;
    cout7 = (((A >> 7)&1)&((B >> 7)&1))|(xout7 & cout6);
    xout8 = ((A >> 8)&1)^((B >> 8)&1);
    dout8 = xout8 ^ cout7;
    cout8 = (((A >> 8)&1)&((B >> 8)&1))|(xout8 & cout7);
    xout9 = ((A >> 9)&1)^((B >> 9)&1);
    dout9 = xout9 ^ cout8;
    cout9 = (((A >> 9)&1)&((B >> 9)&1))|(xout9 & cout8);
    xout10 = ((A >> 10)&1)^((B >> 10)&1);
    dout10 = xout10 ^ cout9;
    cout10 = (((A >> 10)&1)&((B >> 10)&1))|(xout10 & cout9);
    xout11 = ((A >> 11)&1)^((B >> 11)&1);
    dout11 = xout11 ^ cout10;
    cout11 = (((A >> 11)&1)&((B >> 11)&1))|(xout11 & cout10);
    xout12 = ((A >> 12)&1)^((B >> 12)&1);
    dout12 = xout12 ^ cout11;
    cout12 = (((A >> 12)&1)&((B >> 12)&1))|(xout12 & cout11);
    xout13 = ((A >> 13)&1)^((B >> 13)&1);
    dout13 = xout13 ^ cout12;
    cout13 = (((A >> 13)&1)&((B >> 13)&1))|(xout13 & cout12);
    xout14 = ((A >> 14)&1)^((B >> 14)&1);
    dout14 = xout14 ^ cout13;
    cout14 = (((A >> 14)&1)&((B >> 14)&1))|(xout14 & cout13);
    xout15 = ((A >> 15)&1)^((B >> 15)&1);
    dout15 = xout15 ^ cout14;
    //cout15 = (((A >> 15)&1)&((B >> 15)&1))|(((cout14 >> 0)&1)&((xout15 >> 0)&1));


    //O = 0;
    O[0] = dout0;
    O[1] = dout1;
    O[2] = dout2;
    O[3] = dout3;
    O[4] = dout4;
    O[5] = dout5;
    O[6] = dout6;
    O[7] = dout7;
    O[8] = dout8;
    O[9] = dout9;
    O[10] = dout10;
    O[11] = dout11;
    O[12] = dout12;
    O[13] = dout13;
    O[14] = dout14;
    O[15] = dout15;
    //O |= (cout15&1) << 16;
    return O;
}

