
/*
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: X11
*/

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "hls_stream.h"
#include "ap_int.h"
#include <stdlib.h>
#include "ap_int.h"

#define SIZE 512 * 512

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


extern "C" {

void canny_"version"(unsigned char* dstimg, unsigned char* srcimg, uint8_t width, uint8_t height) {

#pragma HLS INTERFACE m_axi port=dstimg offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=srcimg offset=slave bundle=gmem1

    int w_ = width;
    int h_ = height;
    unsigned char src[SIZE];
    unsigned char dst[SIZE];
    unsigned char dst_h[SIZE];
    unsigned char G_[SIZE];
    unsigned char M_[SIZE];
    unsigned char s_[SIZE];
    int t_[SIZE];

    int offset_xy = 1;  // for kernel = 3

    for_src: for (int xy = 0; xy < SIZE; xy++)
#pragma HLS PIPELINE II=1
        src[xy] = srcimg[xy];

    // gaussian filter
    for1_gaus: for (int x = 0; x < w_; x++) {
#pragma HLS PIPELINE II=1
    	for2_gaus: for (int y = 0; y < h_; y++) {
            int pos = x + (y * w_);
            if (x < offset_xy || x >= (w_ - offset_xy) || y < offset_xy ||
                y >= (h_ - offset_xy)) {
                dst[pos] = src[pos];
            } else {
                int x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, sum;

                x1 = add16se_"add1"(src[(x - 1) + ((y - 1) * w_)], src[(x + 1) + ((y - 1) * w_)]);
                x2 = mul16s_"mul1"(src[x + ((y - 1) * w_)], 2);
                x3 = add16se_"add2"(x1, x2);
                x4 = mul16s_"mul2"(src[(x - 1) + (y * w_)], 2);
                x5 = mul16s_"mul3"(src[x + (y * w_)], 4);
                x6 = mul16s_"mul4"(src[(x + 1) + (y * w_)], 2);
                x7 = add16se_"add3"(x4, x5);
                x8 = add16se_"add4"(x6, x7);
                x9 = add16se_"add5"(src[(x - 1) + ((y + 1) * w_)], src[(x + 1) + ((y + 1) * w_)]);
                x10 = mul16s_"mul5"(src[x + ((y + 1) * w_)], 2);
                x11 = add16se_"add6"(x9, x10);
                x12 = add16se_"add7"(x3, x8);
                sum = add16se_"add8"(x11, x12);

                sum /= 16;
                sum = (sum < 0) ? 0 : sum;
                sum = (sum > 255) ? 255 : sum;

                dst[pos] = (unsigned char) sum;
            }
        }
    }

    // apply sobel kernels
    for1_sobel: for (int x = offset_xy; x < w_ - offset_xy; x++) {
#pragma HLS PIPELINE II=4
    	for2_sobel: for (int y = offset_xy; y < h_ - offset_xy; y++) {
            int src_pos = x + (y * w_);
            int sobel, theta;
            int x1, x2, x3, x4, x5, x6;
            int gx_sum, gy_sum;

            x1 = add16se_"add9"(dst[(y - 1) * w_ + (x - 1)], dst[(y + 1) * w_ + (x - 1)]);
            x2 = mul16s_"mul6"(dst[y * w_ + (x - 1)], 2);
            x3 = add16se_"add10"(x1, x2);
            x4 = add16se_"add11"(dst[(y - 1) * w_ + (x + 1)], dst[(y + 1) * w_ + (x + 1)]);
            x5 = mul16s_"mul7"(dst[y * w_ + (x + 1)], 2);
            x6 = add16se_"add12"(x4, x5);
            gx_sum = x3 - x6;
            gx_sum = (gx_sum < 0) ? -gx_sum : gx_sum;

            x1 = add16se_"add13"(dst[(y - 1) * w_ + (x - 1)], dst[(y - 1) * w_ + (x + 1)]);
            x2 = mul16s_"mul8"(dst[(y - 1) * w_ + x], 2);
            x3 = add16se_"add14"(x1, x2);
            x4 = add16se_"add15"(dst[(y + 1) * w_ + (x - 1)], dst[(y + 1) * w_ + (x + 1)]);
            x5 = mul16s_"mul9"(dst[(y + 1) * w_ + x], 2);
            x6 = add16se_"add16"(x4, x5);
            gy_sum = x3 - x6;
            gy_sum = (gy_sum < 0) ? -gy_sum : gy_sum;

            if (gx_sum == 0 || gy_sum == 0) {
                sobel = 0;
                theta = 0;
            } else {
                sobel = ((gx_sum + gy_sum) > 255) ? 255 : (gx_sum + gy_sum);
                theta = gx_sum * 256 / gy_sum;
            }
            G_[src_pos] = sobel;
            t_[src_pos] = theta;
        }
    }

    // gradient hypot & direction
    for1_grad: for (int x = offset_xy; x < w_ - offset_xy; x++) {
#pragma HLS PIPELINE II=1
    	for2_grad: for (int y = offset_xy; y < h_ - offset_xy; y++) {
            int src_pos = x + (y * w_);
            int segment = 0;
            int theta = t_[src_pos];
            if (theta != 0) {
                if ((theta <= 22 && theta >= -22) || (theta <= -157) ||
                    (theta >= 157)) {
                    segment = 1;  // "-"
                } else if ((theta > 22 && theta <= 67) ||
                           (theta > -157 && theta <= -112)) {
                    segment = 2;  // "/"
                } else if ((theta > 67 && theta <= 112) ||
                           (theta >= -112 && theta < -67)) {
                    segment = 3;  // "|"
                } else if ((theta >= -67 && theta < -22) ||
                           (theta > 112 && theta < 157)) {
                    segment = 4;  // "\"
                }
            }
            s_[src_pos] = (unsigned char)segment;
        }
    }

    // local maxima: non maxima suppression
    for1_nms: for (int x = 1; x < w_ - 1; x++) {
#pragma HLS PIPELINE II=1
    	for2_nms: for (int y = 1; y < h_ - 1; y++) {
            int pos = x + (y * w_);
            int pix = G_[pos];
            if (s_[pos] == 1) {
                if (G_[pos - 1] >= G_[pos] || G_[pos + 1] > G_[pos]) {
                    pix = 0;
                }
            } else if (s_[pos] == 2) {
                if (G_[pos - (w_ - 1)] >= G_[pos] || G_[pos + (w_ - 1)] > G_[pos]) {
                    pix = 0;
                }
            } else if (s_[pos] == 3) {
                if (G_[pos - (w_)] >= G_[pos] || G_[pos + (w_)] > G_[pos]) {
                    pix = 0;
                }
            } else if (s_[pos] == 4) {
                if (G_[pos - (w_ + 1)] >= G_[pos] || G_[pos + (w_ + 1)] > G_[pos]) {
                    pix = 0;
                }
            } else {
                pix = 0;
            }
            M_[pos] = pix;
        }
    }

    // double threshold
    for1_dt: for (int x = 0; x < w_; x++) {
#pragma HLS PIPELINE II=1
    	for2_dt: for (int y = 0; y < h_; y++) {
            int src_pos = x + (y * w_);
            int pix;
            if (M_[src_pos] > 90) {
                pix = 255;
            } else if (M_[src_pos] > 20) {
                pix = 100;
            } else {
                pix = 0;
            }
            dst[src_pos] = pix;
        }
    }

    // edges with hysteresis
    for1_he: for (int x = 1; x < w_ - 1; x++) {
#pragma HLS PIPELINE II=3
    	for2_he: for (int y = 1; y < h_ - 1; y++) {
            int src_pos = x + (y * w_);
            int pix;
            if (dst[src_pos] == 255) {
                pix = 255;
            } else if (dst[src_pos] == 100) {
                if (dst[src_pos - 1] == 255 || dst[src_pos + 1] == 255 ||
                    dst[src_pos - 1 - w_] == 255 ||
                    dst[src_pos + 1 - w_] == 255 || dst[src_pos + w_] == 255 ||
                    dst[src_pos + w_ - 1] == 255 ||
                    dst[src_pos + w_ + 1] == 255) {
                    pix = 255;
                } else {
                    pix = 0;
                }
            } else {
                pix = 0;
            }
            dst_h[src_pos] = pix;
        }
    }

    for_dst: for (int xy = 0; xy < SIZE; xy++)
#pragma HLS PIPELINE II=1
        dstimg[xy] = dst_h[xy];
}

}
