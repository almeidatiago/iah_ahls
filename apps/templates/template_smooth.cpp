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

#define FILTER_V_SIZE		5
#define FILTER_H_SIZE		5

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

ap_int<16> filter(const window w) {

    ap_int<16> x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20,
            x21, x22, x23, sum;
#pragma HLS pipeline II=1
    x1 = add16se_"add1"(w.pix[0][0], w.pix[0][1]);
    x2 = add16se_"add1"(w.pix[0][2], w.pix[0][3]);
    x3 = add16se_"add1"(x1, w.pix[0][4]);
    x4 = add16se_"add1"(x2, x3);

    x5 = add16se_"add2"(w.pix[1][0], w.pix[1][1]);
    x6 = add16se_"add2"(w.pix[1][2], w.pix[1][3]);
    x7 = add16se_"add2"(x5, w.pix[1][4]);
    x8 = add16se_"add2"(x6, x7);
    
    x9 = add16se_"add3"(w.pix[2][0], w.pix[2][1]);
    x10 = add16se_"add3"(w.pix[2][2], w.pix[2][3]);
    x11 = add16se_"add3"(x9, w.pix[2][4]);
    x12 = add16se_"add3"(x10, x11);
    
    x13 = add16se_"add4"(w.pix[3][0], w.pix[3][1]);
    x14 = add16se_"add4"(w.pix[3][2], w.pix[3][3]);
    x15 = add16se_"add4"(x13, w.pix[3][4]);
    x16 = add16se_"add4"(x14, x15);
    
    x17 = add16se_"add5"(w.pix[4][0], w.pix[4][1]);
    x18 = add16se_"add5"(w.pix[4][2], w.pix[4][3]);
    x19 = add16se_"add5"(x17, w.pix[4][4]);
    x20 = add16se_"add5"(x18, x19);
    
    x21 = add16se_"add6"(x4, x8);
    x22 = add16se_"add6"(x12, x16);
    x23 = add16se_"add6"(x21, x22);
    sum = add16se_"add6"(x23, x20);
    
    sum = sum / (ap_int<16>)25;

	return MAX((ap_int<16>)0, MIN((ap_int<16>)255, sum));
}


void Filter2D(
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
				sum = filter(w);;
			}
            // Normalize result
            unsigned char outpix = sum;

            // Write the output pixel
            pixel_stream.write(outpix);
        }
    }
}


extern "C" {

void smooth(
        const unsigned char  src[MAX_IMAGE_WIDTH*MAX_IMAGE_HEIGHT],
        unsigned char        dst[MAX_IMAGE_WIDTH*MAX_IMAGE_HEIGHT], 
        unsigned short height, 
        unsigned short width)
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
	Filter2D(width, height, window_stream, output_stream);

	// Write incoming stream of pixels and write them to global memory over AXI4 MM
	WriteToMem(width, height, width, output_stream, dst);
  }

}


