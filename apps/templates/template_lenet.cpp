#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

//#include "lenet.h"
//#include "mnist_file.h"
#include "cnn_defines.h"
#include "MiniLenetq_weights_and_bias_all_layers.h"

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

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
//#define UINT8_MIN 0
//#define UINT8_MAX 255

// Simple requantization function
uint8_t requantize(int32_t acc, uint32_t m_scale, int8_t zero_point) {
	int64_t scaled_sum;
	int32_t output_val;
//#pragma HLS BIND_OP variable=scaled_sum op=mul impl=dsp
//#pragma HLS BIND_OP variable=output_val op=add impl=dsp
    // Apply scaling to accumulated sum
    scaled_sum = (int64_t)(m_scale) * (int64_t)(acc);
    output_val = (int32_t)((scaled_sum + (1LL << 31)) >> 32);
    // Add zero-point offset if needed
    output_val += (int32_t)(zero_point);
    // Clamp result to int8_t range
    return (uint8_t)(MAX(0, MIN(output_val, 255)));
}

void padding_1(const int8_t features[IMAGE_FEATURES][IMAGE_HEIGHT][IMAGE_WIDTH],
               uint8_t padded_features[IMAGE_FEATURES][PADDED_IMAGE_HEIGHT][PADDED_IMAGE_WIDTH]) {
#pragma HLS INLINE off
    // For each feature
    pad1_F: for (int f = 0; f < IMAGE_FEATURES; f++) {
        // Go through rows and columns
        pad1_H: for (int h = 0; h < IMAGE_HEIGHT + 2*PADDING ; h++) {
            pad1_W: for (int w = 0; w < IMAGE_WIDTH + 2*PADDING; w++) {
#pragma HLS PIPELINE II=1
                // Fill left and right columns with 0.0
                // Fill top and bottom rows with 0.0
                if (w < PADDING || w > IMAGE_WIDTH + PADDING -1 ||
                    h < PADDING || h > IMAGE_HEIGHT + PADDING -1) {
                    padded_features[f][h][w] = 17;

                }
                    // Fill the rest of the image with the actual pixel
                else {
                    padded_features[f][h][w]=(uint8_t)features[f][h-PADDING][w-PADDING];
                }
            }
        }
    }
}

// Applies a 2D Convolution according to: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
void convolution2DRelu_1(const int8_t input_kernels [CONV1_KERNELS][CONV1_FEATURES][CONV1_KERNEL_HEIGHT][CONV1_KERNEL_WIDTH],
                         const int32_t bias[CONV1_KERNELS],
                         const uint8_t input_features [CONV1_FEATURES][CONV1_FEATURE_HEIGHT][CONV1_FEATURE_WIDTH],
                         uint8_t output_features [CONV1_KERNELS][CONV1_CONVOLVED_FEATURE_HEIGHT][CONV1_CONVOLVED_FEATURE_WIDTH]) {
    int32_t accumulated;
    // For each kernel
    conv1_K: for (int k = 0; k < CONV1_KERNELS; k++) {
    // Go through features rows and columns
        conv1_FH: for (int fh = 0; fh < CONV1_CONVOLVED_FEATURE_HEIGHT; fh++) {
            conv1_FW: for (int fw = 0; fw < CONV1_CONVOLVED_FEATURE_WIDTH; fw++) {
#pragma HLS PIPELINE II=1
            // Reset accumulated value
                accumulated = 0;
                // Go through the kernel rows and columns
                conv1_KH: for (int kh = 0; kh < CONV1_KERNEL_HEIGHT; kh++) {
                    conv1_KW: for (int kw = 0; kw < CONV1_KERNEL_WIDTH; kw++) {
                    // Convolve each feature with the corresponding kernel and add the result
                        conv1_F: for (int f = 0; f < CONV1_FEATURES; f++) {
#pragma HLS EXPRESSION_BALANCE
                        	int16_t op1 = add16se_"add1"(input_features[f][fh + kh][fw + kw], - ZERO_IMAGE[f]);
                        	int16_t op2 = input_kernels[k][f][kh][kw];
                        	accumulated += mul16s_"mul1"(op1, op2);
                        }
                    }
                }
                output_features[k][fh][fw] = requantize((accumulated + bias[k]), M_CONV1[k], ZERO_CONV_1);
            }
        }
    }
}

void maxPool_1(const uint8_t input_features[CONV1_KERNELS][CONV1_CONVOLVED_FEATURE_HEIGHT][CONV1_CONVOLVED_FEATURE_WIDTH],
               uint8_t output_features[CONV1_KERNELS][CONV1_MAXPOOL_FINAL_HEIGHT][CONV1_MAXPOOL_FINAL_WIDTH]) {
#pragma HLS INLINE off
    int8_t maxValue;
    // For each final feature
    max1_F: for (int f = 0; f < CONV1_KERNELS; f++) {
    // Go through final CONV1_KERNELS rows and columns
        max1_HM: for (int hm = 0; hm < CONV1_MAXPOOL_FINAL_HEIGHT; hm++) {
            max1_WM: for (int wm = 0; wm < CONV1_MAXPOOL_FINAL_WIDTH; wm++) {
#pragma HLS PIPELINE II=1
                maxValue = 0;
                // Go through final CONV1_KERNELS rows and columns according to the selected step
                max1_HF: for (int hf = 0; hf < CONV1_MAXPOOL_STEP; hf++) {
                    max1_WF: for (int wf = 0; wf < CONV1_MAXPOOL_STEP; wf++) {
                    // Determine the maximum value within this region
                        if (input_features[f][hm * CONV1_MAXPOOL_STEP + hf][wm* CONV1_MAXPOOL_STEP + wf] > maxValue) {
                            maxValue = input_features[f][hm * CONV1_MAXPOOL_STEP + hf][wm * CONV1_MAXPOOL_STEP + wf];
                        }
                    }
                }
                // Assign the maximum value found to the final feature
                output_features[f][hm][wm] = maxValue;
            }
        }
    }

}

void convolution2DRelu_2(const int8_t input_kernels [CONV2_KERNELS][CONV1_KERNELS][CONV2_KERNEL_HEIGHT][CONV2_KERNEL_WIDTH],
                         const int32_t bias[CONV2_KERNELS],
                         const uint8_t input_features [CONV1_KERNELS][PADDED_FEATURES_WIDTH][PADDED_FEATURES_HEIGHT],
                         uint8_t output_features [CONV2_KERNELS][CONV2_CONVOLVED_FEATURE_HEIGHT][CONV2_CONVOLVED_FEATURE_WIDTH]) {
    int32_t accumulated;
    // For each kernel
    conv2_K: for (int k = 0; k < CONV2_KERNELS; k++) {
    // Go through features rows and columns
        conv2_FH: for (int fh = 0; fh < CONV2_CONVOLVED_FEATURE_HEIGHT; fh++) {
            conv2_FW: for (int fw = 0; fw < CONV2_CONVOLVED_FEATURE_WIDTH; fw++) {
#pragma HLS PIPELINE II=13
                // Reset accumulated value
                accumulated = 0;
                // Go through the kernel rows and columns
                conv2_KH: for (int kh = 0; kh < CONV2_KERNEL_HEIGHT; kh++) {
//#pragma HLS PIPELINE II=1
                    conv2_KW: for (int kw = 0; kw < CONV2_KERNEL_WIDTH; kw++) {
                    // Convolve each feature with the corresponding kernel and add the result
                        conv2_F: for (int f = 0; f < CONV1_KERNELS; f++) {
#pragma HLS EXPRESSION_BALANCE
							accumulated += mul16s_"mul2"(input_features[f][fh + kh][fw + kw], input_kernels[k][f][kh][kw]);
                        }
                    }
                }
                output_features[k][fh][fw] = requantize((accumulated + bias[k]), M_CONV2[k], ZERO_CONV_2);
            }
        }
    }
}

// Add 0.0 as padding on the top and bottom rows and left and right columns of provided features
void padding_2(const uint8_t features[CONV1_KERNELS][CONV1_MAXPOOL_FINAL_HEIGHT][CONV1_MAXPOOL_FINAL_WIDTH],
               uint8_t padded_features[CONV1_KERNELS][PADDED_FEATURES_HEIGHT][PADDED_FEATURES_WIDTH]) {
#pragma HLS INLINE off
    // For each feature
    pad2_F: for (int f = 0; f < CONV1_KERNELS; f++) {
    // Go through rows and columns
        pad2_H: for (int h = 0; h < CONV1_MAXPOOL_FINAL_HEIGHT + 2*PADDING ; h++) {
            pad2_W: for (int w = 0; w < CONV1_MAXPOOL_FINAL_WIDTH + 2*PADDING; w++) {
#pragma HLS PIPELINE II=1
                // Fill left and right columns with 0.0
                // Fill top and bottom rows with 0.0
                if (w < PADDING || w > CONV1_MAXPOOL_FINAL_WIDTH + PADDING -1 ||
                    h < PADDING || h > CONV1_MAXPOOL_FINAL_HEIGHT + PADDING -1) {
                    padded_features[f][h][w] = 0;
                }
                    // Fill the rest of the image with the actual pixel
                else {
                    padded_features[f][h][w]=features[f][h-PADDING][w-PADDING];
                }
            }
        }
    }
}

void maxPool_2(const uint8_t input_features [CONV2_KERNELS][CONV2_CONVOLVED_FEATURE_HEIGHT][CONV2_CONVOLVED_FEATURE_WIDTH],
               uint8_t output_features [CONV2_KERNELS][CONV2_MAXPOOL_FINAL_WIDTH][CONV2_MAXPOOL_FINAL_HEIGHT]) {
#pragma HLS INLINE off
    int8_t maxValue;
    // For each final feature
    max2_F: for (int f = 0; f < CONV2_KERNELS; f++) {
    // Go through final CONV1_KERNELS rows and columns
        max2_HM: for (int hm = 0; hm < CONV2_MAXPOOL_FINAL_HEIGHT; hm++) {
            max2_WM: for (int wm = 0; wm < CONV2_MAXPOOL_FINAL_WIDTH; wm++) {
#pragma HLS PIPELINE II=1
                maxValue = 0;
                // Go through final CONV1_KERNELS rows and columns according to the selected step
                max2_HF: for (int hf = 0; hf < CONV2_MAXPOOL_STEP; hf++) {
                    max2_WF: for (int wf = 0; wf < CONV2_MAXPOOL_STEP; wf++) {
                    // Determine the maximum value within this region
                        if (input_features[f][hm * CONV2_MAXPOOL_STEP + hf][wm* CONV2_MAXPOOL_STEP + wf] > maxValue) {
                            maxValue = input_features[f][hm * CONV2_MAXPOOL_STEP + hf][wm * CONV2_MAXPOOL_STEP + wf];
                        }
                    }
                }
                // Assign the maximum value found to the final feature
                output_features[f][hm][wm] = maxValue;
            }
        }
    }
}

void flattenLayer(const uint8_t input_features [CONV2_KERNELS][CONV2_MAXPOOL_FINAL_WIDTH][CONV2_MAXPOOL_FINAL_HEIGHT],
                  uint8_t output_features[FC1_INPUT_FEATURES]) {
#pragma HLS INLINE off
    // For each feature
    flat_F: for (int f = 0; f < CONV2_KERNELS; f++) {
        // Go through features rows and columns
        flat_H: for (int h = 0; h < CONV2_MAXPOOL_FINAL_HEIGHT; h++){
            flat_W: for (int w = 0; w < CONV2_MAXPOOL_FINAL_WIDTH; w++){
#pragma HLS PIPELINE II=1
                // Copy each value into a unidimensional array
                output_features[f*CONV2_MAXPOOL_FINAL_HEIGHT*CONV2_MAXPOOL_FINAL_WIDTH+ h*CONV2_MAXPOOL_FINAL_WIDTH+ w] = input_features[f][h][w];
            }
        }
    }
}

void fullyConnected_1(const uint8_t input_features [FC1_INPUT_FEATURES],
                      const int8_t input_weights [FC1_FEATURES][FC1_INPUT_FEATURES],
                      const int32_t bias[FC1_FEATURES],
                      uint8_t output_features[FC1_FEATURES]) {
    int32_t accumulated;
    // Go through all the values of that feature
    fc1_F: for (int f = 0; f < FC1_FEATURES; f++) {
//#pragma HLS PIPELINE II=1
		accumulated = 0;
		// For each feature
		fc1_NIF: for (int nif = 0; nif < FC1_INPUT_FEATURES; nif++) {
#pragma HLS EXPRESSION_BALANCE
			accumulated += mul16s_"mul3"(input_features[nif], input_weights[f][nif]);
		}
		output_features[f] = requantize((accumulated + bias[f]), M_FC1[f], ZERO_FC1);
    }
}

void fullyConnected_2(const uint8_t input_features [FC1_FEATURES],
                      const int8_t input_weights [FC2_FEATURES][FC1_FEATURES],
                      const int32_t bias[FC2_FEATURES],
                      uint8_t output_features[FC2_FEATURES]) {
    int32_t accumulated;
    // Go through all the values of that feature
    fc2_F: for (int f = 0; f < FC2_FEATURES; f++) {
//#pragma HLS PIPELINE II=1
        accumulated = 0;
        // For each feature
        fc2_NIF: for (int nif = 0; nif < FC1_FEATURES; nif++) {
#pragma HLS EXPRESSION_BALANCE
        	int16_t op1 = add16se_"add2"(ap_int<16>(input_features[nif]), -ap_int<16>(ZERO_FC1));
			int16_t op2 = input_weights[f][nif];
        	accumulated += mul16s_"mul4"(op1, op2);
        }
        output_features[f] = requantize((accumulated + bias[f]), M_FC2[f], ZERO_FC2);
    }
}

extern "C" {
void lenet_"version"(int8_t input_image[IMAGE_FEATURES][IMAGE_HEIGHT][IMAGE_WIDTH],
		uint8_t output[LAST_LAYER_FEATURES]) {

#pragma HLS INTERFACE mode=m_axi port=input_image depth=784 bundle=gmem0 offset=slave
#pragma HLS INTERFACE mode=m_axi port=output depth=10 bundle=gmem0 offset=slave

	int8_t sample_image[IMAGE_FEATURES][IMAGE_HEIGHT][IMAGE_WIDTH];
    uint8_t padded_image[IMAGE_FEATURES][PADDED_IMAGE_HEIGHT][PADDED_IMAGE_WIDTH];
    uint8_t first_convolution_relu[CONV1_KERNELS][CONV1_CONVOLVED_FEATURE_HEIGHT][CONV1_CONVOLVED_FEATURE_WIDTH];
    uint8_t first_convolution_max[CONV1_KERNELS][CONV1_MAXPOOL_FINAL_HEIGHT][CONV1_MAXPOOL_FINAL_WIDTH];

    uint8_t first_convolution_padded[CONV1_KERNELS][PADDED_FEATURES_WIDTH][PADDED_FEATURES_HEIGHT];
    uint8_t second_convolution_relu[CONV2_KERNELS][CONV2_CONVOLVED_FEATURE_HEIGHT][CONV2_CONVOLVED_FEATURE_WIDTH];
    uint8_t second_convolution_max[CONV2_KERNELS][CONV2_MAXPOOL_FINAL_WIDTH][CONV2_MAXPOOL_FINAL_HEIGHT];

    uint8_t flatten[FC1_INPUT_FEATURES];
    uint8_t fc1[FC1_FEATURES];
    uint8_t fc2[LAST_LAYER_FEATURES];

    up_in_i: for (int i = 0; i < 28; i++)
    	up_in_j: for (int j = 0; j < 28; j++)
#pragma HLS PIPELINE II=1
    		sample_image[0][i][j] = input_image[0][i][j];

    padding_1(sample_image, padded_image);
    convolution2DRelu_1(KERNELQ_CONV_1, BIAS_CONV1, padded_image, first_convolution_relu);
    maxPool_1(first_convolution_relu, first_convolution_max);

    padding_2(first_convolution_max, first_convolution_padded);
    convolution2DRelu_2(KERNELQ_CONV_2, BIAS_CONV2, first_convolution_padded, second_convolution_relu);
    maxPool_2(second_convolution_relu, second_convolution_max);

    flattenLayer(second_convolution_max, flatten);
    fullyConnected_1(flatten, WEIGHTSQ_FC1, BIAS_FC1, fc1);
    fullyConnected_2(fc1, WEIGHTSQ_FC2, BIAS_FC2, fc2);

    up_out: for (int i = 0; i < 10; i++)
#pragma HLS PIPELINE II=1
    	output[i] = fc2[i];

}
}
