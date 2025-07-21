/*
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: X11
*/

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "hls_stream.h"
#include "ap_int.h"

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

#define MAX_IMAGE_WIDTH     4096
#define MAX_IMAGE_HEIGHT    4096

#define FILTER_V_SIZE		7
#define FILTER_H_SIZE		7

#ifndef MIN
#define MIN(a,b) ((a<b)?a:b)
#endif

#ifndef MAX
#define MAX(a,b) ((a>b)?a:b)
#endif

typedef unsigned char      		U8;
typedef unsigned short     		U16;
typedef unsigned int       		U32;

typedef signed char        		I8;
typedef signed short       		I16;
typedef signed int         		I32;

void ReadFromMem(
        unsigned short       width,
        unsigned short       height,
        unsigned short       stride,
        const unsigned char *src,
        hls::stream<U8>     &pixel_stream )
{
    stride = (stride/64)*64; // Makes compiler see that stride is a multiple of 64, enables auto-widening
    unsigned offset = 0;
    unsigned x = 0;
    read_image: for (int n = 0; n < height*stride; n++) {
        U8 pix = src[n];
        if (x<width) pixel_stream.write( pix );
        if (x==(stride-1)) x=0; else x++;
     }
}


void WriteToMem(
        unsigned short       width,
        unsigned short       height,
        unsigned short       stride,
        hls::stream<U8>     &pixel_stream,
        unsigned char       *dst)
{
    assert(stride <= MAX_IMAGE_WIDTH);
    assert(height <= MAX_IMAGE_HEIGHT);
    assert(stride%64 == 0);

    stride = (stride/64)*64; // Makes compiler see that stride is a multiple of 64, enables auto-widening
    unsigned offset = 0;
    unsigned x = 0;
    write_image: for (int n = 0; n < height*stride; n++) {
        U8 pix = (x<width) ? pixel_stream.read() : 0;
        dst[n] = pix;
        if (x==(stride-1)) x=0; else x++;
    }
}


struct window {
    U8 pix[FILTER_V_SIZE][FILTER_H_SIZE];
};


void Window2D(
        unsigned short        width,
        unsigned short        height,
        hls::stream<U8>      &pixel_stream,
        hls::stream<window>  &window_stream)
{
    // Line buffers - used to store [FILTER_V_SIZE-1] entire lines of pixels
    U8 LineBuffer[FILTER_V_SIZE-1][MAX_IMAGE_WIDTH];
#pragma HLS ARRAY_PARTITION variable=LineBuffer dim=1 complete
#pragma HLS DEPENDENCE variable=LineBuffer inter false
#pragma HLS DEPENDENCE variable=LineBuffer intra false

    // Sliding window of [FILTER_V_SIZE][FILTER_H_SIZE] pixels
    window Window;

    unsigned col_ptr = 0;
    unsigned ramp_up = width*((FILTER_V_SIZE-1)/2)+(FILTER_H_SIZE-1)/2;
    unsigned num_pixels = width*height;
    unsigned num_iterations = num_pixels + ramp_up;

    const unsigned max_iterations = MAX_IMAGE_WIDTH*MAX_IMAGE_HEIGHT + MAX_IMAGE_WIDTH*((FILTER_V_SIZE-1)/2)+(FILTER_H_SIZE-1)/2;

    // Iterate until all pixels have been processed
    update_window: for (int n=0; n<num_iterations; n++)
    {
#pragma HLS LOOP_TRIPCOUNT max=max_iterations
#pragma HLS PIPELINE II=1

        // Read a new pixel from the input stream
        U8 new_pixel = (n<num_pixels) ? pixel_stream.read() : 0;

        // Shift the window and add a column of new pixels from the line buffer
        for(int i = 0; i < FILTER_V_SIZE; i++) {
            for(int j = 0; j < FILTER_H_SIZE-1; j++) {
                Window.pix[i][j] = Window.pix[i][j+1];
            }
            Window.pix[i][FILTER_H_SIZE-1] = (i<FILTER_V_SIZE-1) ? LineBuffer[i][col_ptr] : new_pixel;
        }

        // Shift pixels in the column of pixels in the line buffer, add the newest pixel
        for(int i = 0; i < FILTER_V_SIZE-2; i++) {
            LineBuffer[i][col_ptr] = LineBuffer[i+1][col_ptr];
        }
        LineBuffer[FILTER_V_SIZE-2][col_ptr] = new_pixel;

        // Update the line buffer column pointer
        if (col_ptr==(width-1)) {
            col_ptr = 0;
        } else {
            col_ptr++;
        }

        // Write output only when enough pixels have been read the buffers and ramped-up
        if (n>=ramp_up) {
            window_stream.write(Window);
        }

    }
}

ap_int<16> filter(const window w, ap_int<16> m1, ap_int<16> m2, ap_int<16> m3, ap_int<16> m4) {
    
    ap_int<16> x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20,
            x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40,
            x41, x42, x43, x44, x45, x46, x47, x48, x49, x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, sum;

#pragma HLS pipeline II=1
    x1 = mul16s_"mul1"(w.pix[3][3], m1);
    x2 = mul16s_"mul2"(w.pix[2][2], m2);
    x3 = mul16s_"mul2"(w.pix[2][3], m2);
    x4 = mul16s_"mul2"(w.pix[2][4], m2);
    x5 = mul16s_"mul2"(w.pix[3][2], m2);
    x6 = mul16s_"mul2"(w.pix[3][4], m2);
    x7 = mul16s_"mul2"(w.pix[4][2], m2);
    x8 = mul16s_"mul2"(w.pix[4][3], m2);
    x9 = mul16s_"mul2"(w.pix[4][4], m2);
    x10 = mul16s_"mul3"(w.pix[1][2], m3);
    x11 = mul16s_"mul3"(w.pix[1][3], m3);
    x12 = mul16s_"mul3"(w.pix[1][4], m3);
    x13 = mul16s_"mul3"(w.pix[2][1], m3);
    x14 = mul16s_"mul3"(w.pix[2][5], m3);
    x15 = mul16s_"mul3"(w.pix[3][1], m3);
    x16 = mul16s_"mul3"(w.pix[3][5], m3);
    x17 = mul16s_"mul3"(w.pix[4][1], m3);
    x18 = mul16s_"mul3"(w.pix[4][5], m3);
    x19 = mul16s_"mul3"(w.pix[5][2], m3);
    x20 = mul16s_"mul3"(w.pix[5][3], m3);
    x21 = mul16s_"mul3"(w.pix[5][4], m3);
    x22 = mul16s_"mul4"(w.pix[1][1], m4);
    x23 = mul16s_"mul4"(w.pix[1][5], m4);
    x24 = mul16s_"mul4"(w.pix[5][1], m4);
    x25 = mul16s_"mul4"(w.pix[5][5], m4);
    
    x26 = add16se_"add1"(x1, x2);
    x27 = add16se_"add1"(x3, x4);
    x28 = add16se_"add1"(x5, x6);
    x29 = add16se_"add1"(x7, x8);
    x30 = add16se_"add1"(x9, x10);
    x31 = add16se_"add2"(x11, x12);
    x32 = add16se_"add2"(x13, x14);
    x33 = add16se_"add2"(x15, x16);
    x34 = add16se_"add2"(x17, x18);
    x35 = add16se_"add2"(x19, x20);
    x36 = add16se_"add2"(x21, x22);
    x37 = add16se_"add3"(x23, x24);
    x38 = add16se_"add3"(x25, x26);
    x39 = add16se_"add4"(x27, x28);
    x40 = add16se_"add4"(x29, x30);
    x41 = add16se_"add4"(x31, x32);
    x42 = add16se_"add4"(x33, x34);
    x43 = add16se_"add4"(x35, x36);
    x44 = add16se_"add4"(x37, x38);
    x45 = add16se_"add4"(x39, x40);
    x46 = add16se_"add4"(x41, x42);
    x47 = add16se_"add4"(x43, x44);
    x48 = add16se_"add4"(x45, x46);
    x49 = add16se_"add4"(x47, x48);
    
    x50 = x49 - w.pix[0][2];
    x51 = x50 - w.pix[0][3];
    x52 = x51 - w.pix[0][4];
    x53 = x52 - w.pix[2][0];
    x54 = x53 - w.pix[2][6];
    x55 = x54 - w.pix[3][0];
    x56 = x55 - w.pix[3][6];
    x57 = x56 - w.pix[4][0];
    x58 = x57 - w.pix[4][6];
    x59 = x58 - w.pix[6][2];
    x60 = x59 - w.pix[6][3];
    sum = x60 - w.pix[6][4];

	return MAX((ap_int<16>)0, MIN((ap_int<16>)255, sum));
}


void Filter2D(
        ap_int<16> m1, ap_int<16> m2, ap_int<16> m3, ap_int<16> m4,
        unsigned short       width,
        unsigned short       height,
        hls::stream<window> &window_stream,
		hls::stream<U8>     &pixel_stream )
{

    // Process the incoming stream of pixel windows
    apply_filter: for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
#pragma HLS PIPELINE II=1
            // Read a 2D window of pixels
            window w = window_stream.read();

            // Apply filter to the 2D window
            int sum = 0;

            // Set output to 0 if the 5x5 receptive field is out of bound.
			if ((i > 1) & (i < height - 2) & (j > 1) & (j < width - 2)) {
				// Apply the sobel filter at the current "receptive field".
				sum = filter(w, m1, m2, m3, m4);
			}
            // Normalize result
            unsigned char outpix = sum;

            // Write the output pixel
            pixel_stream.write(outpix);
        }
    }
}


extern "C" {

void gaussian(
        const unsigned char  src[MAX_IMAGE_WIDTH*MAX_IMAGE_HEIGHT],
        unsigned char        dst[MAX_IMAGE_WIDTH*MAX_IMAGE_HEIGHT], 
        unsigned short height, 
        unsigned short width, 
        ap_int<16> m1, ap_int<16> m2, ap_int<16> m3, ap_int<16> m4)
  {


#pragma HLS DATAFLOW
	// Stream of pixels from kernel input to filter, and from filter to output
    hls::stream<U8>      pixel_stream;
    hls::stream<window>  window_stream; // Set FIFO depth to 0 to minimize resources
    hls::stream<U8>      output_stream;

	// Read image data from global memory over AXI4 MM, and stream pixels out
    ReadFromMem(width, height, width, src, pixel_stream);

    // Read incoming pixels and form valid HxV windows
    Window2D(width, height, pixel_stream, window_stream);

	// Process incoming stream of pixels, and stream pixels out
	Filter2D(m1, m2, m3, m4, width, height, window_stream, output_stream);

	// Write incoming stream of pixels and write them to global memory over AXI4 MM
	WriteToMem(width, height, width, output_stream, dst);
  }

}




