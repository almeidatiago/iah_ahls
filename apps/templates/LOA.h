#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

typedef char bit;
bit carry = 0;

bit fulladd( bit A, bit B ){
    bit Xor = A ^ B;
    bit ret = carry ^ Xor;
    carry = (carry & Xor) | (A & B);
    return ret;
}

/*
bit afa( bit A, bit B ){
    //bit Xor = (A ^ B);
    //bit ret = (carry | Xor);
    bit ret = (A | B);
    carry = (A & B);
    return ret;
}
*/

int16_t LOA (const uint16_t A, const uint16_t B, uint8_t n) {
    
    int r = 0;
    bit tmp;
            
    for (int i = 0; i < n; ++i) {
        //tmp = afa(((A >> i)&1), ((B >> i)&1));
        tmp = ((A >> i)&1) | ((B >> i)&1);
        r += tmp << i;
    }
    carry = ((A >> (n-1))&1) & ((B >> (n-1))&1);
    
    for (int i = n; i < 16; ++i) {
        tmp = fulladd(((A >> i)&1), ((B >> i)&1));
        r += tmp << i;
    }
            
    return r;

}
