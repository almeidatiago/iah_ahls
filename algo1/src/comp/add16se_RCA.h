#include <stdint.h>
#include <stdlib.h>


int16_t add16se_RCA (const int16_t A, const int16_t B) {
    
    
    int16_t dout0, dout1, dout2, dout3, dout4, dout5, dout6, dout7, dout8, dout9, dout10, dout11, dout12, dout13, dout14, dout15;
    int16_t cout0, cout1, cout2, cout3, cout4, cout5, cout6, cout7, cout8, cout9, cout10, cout11, cout12, cout13, cout14;//, cout15;
    int16_t xout1, xout2, xout3, xout4, xout5, xout6, xout7, xout8, xout9, xout10, xout11, xout12, xout13, xout14, xout15;
    int16_t O;

    dout0 = ((A >> 0)&1)^((B >> 0)&1);
    cout0 = ((A >> 0)&1)&((B >> 0)&1);
    xout1 = ((A >> 1)&1)^((B >> 1)&1);
    dout1 = ((xout1 >> 0)&1)^((cout0 >> 0)&1);
    cout1 = (((A >> 1)&1)&((B >> 1)&1))|(((cout0 >> 0)&1)&((xout1 >> 0)&1));
    xout2 = ((A >> 2)&1)^((B >> 2)&1);
    dout2 = ((xout2 >> 0)&1)^((cout1 >> 0)&1);
    cout2 = (((A >> 2)&1)&((B >> 2)&1))|(((cout1 >> 0)&1)&((xout2 >> 0)&1));
    xout3 = ((A >> 3)&1)^((B >> 3)&1);
    dout3 = ((xout3 >> 0)&1)^((cout2 >> 0)&1);
    cout3 = (((A >> 3)&1)&((B >> 3)&1))|(((cout2 >> 0)&1)&((xout3 >> 0)&1));
    xout4 = ((A >> 4)&1)^((B >> 4)&1);
    dout4 = ((xout4 >> 0)&1)^((cout3 >> 0)&1);
    cout4 = (((A >> 4)&1)&((B >> 4)&1))|(((cout3 >> 0)&1)&((xout4 >> 0)&1));
    xout5 = ((A >> 5)&1)^((B >> 5)&1);
    dout5 = ((xout5 >> 0)&1)^((cout4 >> 0)&1);
    cout5 = (((A >> 5)&1)&((B >> 5)&1))|(((cout4 >> 0)&1)&((xout5 >> 0)&1));
    xout6 = ((A >> 6)&1)^((B >> 6)&1);
    dout6 = ((xout6 >> 0)&1)^((cout5 >> 0)&1);
    cout6 = (((A >> 6)&1)&((B >> 6)&1))|(((cout5 >> 0)&1)&((xout6 >> 0)&1));
    xout7 = ((A >> 7)&1)^((B >> 7)&1);
    dout7 = ((xout7 >> 0)&1)^((cout6 >> 0)&1);
    cout7 = (((A >> 7)&1)&((B >> 7)&1))|(((cout6 >> 0)&1)&((xout7 >> 0)&1));
    xout8 = ((A >> 8)&1)^((B >> 8)&1);
    dout8 = ((xout8 >> 0)&1)^((cout7 >> 0)&1);
    cout8 = (((A >> 8)&1)&((B >> 8)&1))|(((cout7 >> 0)&1)&((xout8 >> 0)&1));
    xout9 = ((A >> 9)&1)^((B >> 9)&1);
    dout9 = ((xout9 >> 0)&1)^((cout8 >> 0)&1);
    cout9 = (((A >> 9)&1)&((B >> 9)&1))|(((cout8 >> 0)&1)&((xout9 >> 0)&1));
    xout10 = ((A >> 10)&1)^((B >> 10)&1);
    dout10 = ((xout10 >> 0)&1)^((cout9 >> 0)&1);
    cout10 = (((A >> 10)&1)&((B >> 10)&1))|(((cout9 >> 0)&1)&((xout10 >> 0)&1));
    xout11 = ((A >> 11)&1)^((B >> 11)&1);
    dout11 = ((xout11 >> 0)&1)^((cout10 >> 0)&1);
    cout11 = (((A >> 11)&1)&((B >> 11)&1))|(((cout10 >> 0)&1)&((xout11 >> 0)&1));
    xout12 = ((A >> 12)&1)^((B >> 12)&1);
    dout12 = ((xout12 >> 0)&1)^((cout11 >> 0)&1);
    cout12 = (((A >> 12)&1)&((B >> 12)&1))|(((cout11 >> 0)&1)&((xout12 >> 0)&1));
    xout13 = ((A >> 13)&1)^((B >> 13)&1);
    dout13 = ((xout13 >> 0)&1)^((cout12 >> 0)&1);
    cout13 = (((A >> 13)&1)&((B >> 13)&1))|(((cout12 >> 0)&1)&((xout13 >> 0)&1));
    xout14 = ((A >> 14)&1)^((B >> 14)&1);
    dout14 = ((xout14 >> 0)&1)^((cout13 >> 0)&1);
    cout14 = (((A >> 14)&1)&((B >> 14)&1))|(((cout13 >> 0)&1)&((xout14 >> 0)&1));
    xout15 = ((A >> 15)&1)^((B >> 15)&1);
    dout15 = ((xout15 >> 0)&1)^((cout14 >> 0)&1);
    //cout15 = (((A >> 15)&1)&((B >> 15)&1))|(((cout14 >> 0)&1)&((xout15 >> 0)&1));


    O = 0;
    O |= (dout0&1) << 0;
    O |= (dout1&1) << 1;
    O |= (dout2&1) << 2;
    O |= (dout3&1) << 3;
    O |= (dout4&1) << 4;
    O |= (dout5&1) << 5;
    O |= (dout6&1) << 6;
    O |= (dout7&1) << 7;
    O |= (dout8&1) << 8;
    O |= (dout9&1) << 9;
    O |= (dout10&1) << 10;
    O |= (dout11&1) << 11;
    O |= (dout12&1) << 12;
    O |= (dout13&1) << 13;
    O |= (dout14&1) << 14;
    O |= (dout15&1) << 15;
    //O |= (cout15&1) << 16;
    return O;
}

