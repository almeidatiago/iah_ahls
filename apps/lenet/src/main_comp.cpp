#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>
#include <chrono>
#include <string.h> 
#include <iostream>
#include <fstream>
using namespace std::chrono;
#include "evaluation.hpp"
#include "confusion.hpp"
#include "mnist_reader.hpp"
#define MNIST_DATA_LOCATION "/home/tiago/pynq-notebooks/apps/lenet/data"
#include "cnn_defines.h"
#include "MiniLenetq_weights_and_bias_all_layers.h"
#include "MiniLenetq_input_images.h"
#include "MiniLenetq_labels_and_predictions.h"
#include "components.h"

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
#define UINT8_MIN 0
#define UINT8_MAX 255

inline int MUL1;
inline int MUL2;
inline int MUL3;
inline int MUL4;
inline int ADD1;
inline int ADD2;
/*inline int ADD3;
inline int ADD4;
inline int ADD5;
inline int ADD6;
inline int ADD7;
inline int ADD8;*/

// Simple requantization function
uint8_t requantize(int32_t acc, uint32_t m_scale, int8_t zero_point) {
    // Apply scaling to accumulated sum
    int64_t scaled_sum = (int64_t)(m_scale) * (int64_t)(acc);
    int32_t output_val = (int32_t)((scaled_sum + (1LL << 31)) >> 32);
    // Add zero-point offset if needed
    output_val += (int32_t)(zero_point);
    // Clamp result to int8_t range
    return (uint8_t)(MAX(UINT8_MIN, MIN(output_val, UINT8_MAX)));
}

void padding_1(const int8_t features[IMAGE_FEATURES][IMAGE_HEIGHT][IMAGE_WIDTH],
               uint8_t padded_features[IMAGE_FEATURES][PADDED_IMAGE_HEIGHT][PADDED_IMAGE_WIDTH]) {

    uint16_t f;
    uint16_t h;
    uint16_t w;
    // For each feature
    pad1_F: for (f = 0; f < IMAGE_FEATURES; f++) {
    // Go through rows and columns
        pad1_H: for (h = 0; h < IMAGE_HEIGHT + 2*PADDING ; h++) {
            pad1_W: for (w = 0; w < IMAGE_WIDTH + 2*PADDING; w++) {
            // Fill left and right columns with 0.0
            // Fill top and bottom rows with 0.0
                if (w < PADDING || w > IMAGE_WIDTH + PADDING -1 ||
                    h < PADDING || h > IMAGE_HEIGHT + PADDING -1) {
                    padded_features[f][h][w] = ZERO_IMAGE[f];

                }
                    // Fill the rest of the image with the actual pixel
                else {
                    padded_features[f][h][w]=features[f][h-PADDING][w-PADDING];
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
    uint16_t k;
    uint16_t fh;
    uint16_t fw;
    uint16_t kh;
    uint16_t kw;
    uint16_t f;
    // For each kernel
    conv1_K: for (k = 0; k < CONV1_KERNELS; k++) {
    // Go through features rows and columns
        conv1_FH: for (fh = 0; fh < CONV1_CONVOLVED_FEATURE_HEIGHT; fh++) {
            conv1_FW: for (fw = 0; fw < CONV1_CONVOLVED_FEATURE_WIDTH; fw++) {
            // Reset accumulated value
                accumulated = 0;
                // Go through the kernel rows and columns
                conv1_KH: for (kh = 0; kh < CONV1_KERNEL_HEIGHT; kh++) {
                    conv1_KW: for (kw = 0; kw < CONV1_KERNEL_WIDTH; kw++) {
                    // Convolve each feature with the corresponding kernel and add the result
                        conv1_F: for (f = 0; f < CONV1_FEATURES; f++) {
                            int16_t op1 = add[ADD1]((int16_t)input_features[f][fh + kh][fw + kw], (int16_t)-ZERO_IMAGE[f]);
                            //int16_t op2 = add[ADD2]((int16_t)input_kernels[k][f][kh][kw], (int16_t)-KERNELQ_ZERO_CONV_1[k]);
                            int16_t op2 = (int16_t)input_kernels[k][f][kh][kw];
                            accumulated +=  mul[MUL1](op1, op2);
                        //accumulated = (int)add[ADD1]((uint16_t)accumulated, (uint16_t)par_acc);
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

    int8_t maxValue;
    uint16_t f;
    uint16_t hm;
    uint16_t wm;
    uint16_t hf;
    uint16_t wf;
    // For each final feature
    max1_F: for (f = 0; f < CONV1_KERNELS; f++) {
    // Go through final CONV1_KERNELS rows and columns
        max1_HM: for (hm = 0; hm < CONV1_MAXPOOL_FINAL_HEIGHT; hm++) {
            max1_WM: for (wm = 0; wm < CONV1_MAXPOOL_FINAL_WIDTH; wm++) {
                maxValue = 0;
                // Go through final CONV1_KERNELS rows and columns according to the selected step
                max1_HF: for (hf = 0; hf < CONV1_MAXPOOL_STEP; hf++) {
                    max1_WF: for (wf = 0; wf < CONV1_MAXPOOL_STEP; wf++) {
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
    uint16_t k;
    uint16_t fh;
    uint16_t fw;
    uint16_t kh;
    uint16_t kw;
    uint16_t f;
    // For each kernel
    conv2_K: for (k = 0; k < CONV2_KERNELS; k++) {
    // Go through features rows and columns
        conv2_FH: for (fh = 0; fh < CONV2_CONVOLVED_FEATURE_HEIGHT; fh++) {
            conv2_FW: for (fw = 0; fw < CONV2_CONVOLVED_FEATURE_WIDTH; fw++) {
            // Reset accumulated value
                accumulated = 0;
                // Go through the kernel rows and columns
                conv2_KH: for (kh = 0; kh < CONV2_KERNEL_HEIGHT; kh++) {
                    conv2_KW: for (kw = 0; kw < CONV2_KERNEL_WIDTH; kw++) {
                    // Convolve each feature with the corresponding kernel and add the result
                        conv2_F: for (f = 0; f < CONV1_KERNELS; f++) {
                            //int16_t op1 = add[ADD3]((int16_t)input_features[f][fh + kh][fw + kw], (int16_t)-ZERO_CONV_1);
                            //int16_t op2 = add[ADD4]((int16_t)input_kernels[k][f][kh][kw], (int16_t)-KERNELQ_ZERO_CONV_2[k]);
                            accumulated += mul[MUL2]((int16_t)input_features[f][fh + kh][fw + kw], (int16_t)input_kernels[k][f][kh][kw]);
                            //accumulated = (int)add[ADD2]((uint16_t)accumulated, (uint16_t)par_acc);
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

    uint16_t f;
    uint16_t h;
    uint16_t w;
    // For each feature
    pad2_F: for (f = 0; f < CONV1_KERNELS; f++) {
    // Go through rows and columns
        pad2_H: for (h = 0; h < CONV1_MAXPOOL_FINAL_HEIGHT + 2*PADDING ; h++) {
            pad2_W: for (w = 0; w < CONV1_MAXPOOL_FINAL_WIDTH + 2*PADDING; w++) {
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
               uint8_t output_features [CONV2_KERNELS][CONV2_MAXPOOL_FINAL_WIDTH][CONV2_MAXPOOL_FINAL_HEIGHT]){

    int8_t maxValue;
    uint16_t f;
    uint16_t hm;
    uint16_t wm;
    uint16_t hf;
    uint16_t wf;
    // For each final feature
    max2_F: for (f = 0; f < CONV2_KERNELS; f++) {
    // Go through final CONV1_KERNELS rows and columns
        max2_HM: for (hm = 0; hm < CONV2_MAXPOOL_FINAL_HEIGHT; hm++) {
            max2_WM: for (wm = 0; wm < CONV2_MAXPOOL_FINAL_WIDTH; wm++) {
                maxValue = 0;
                // Go through final CONV1_KERNELS rows and columns according to the selected step
                max2_HF: for (hf = 0; hf < CONV2_MAXPOOL_STEP; hf++) {
                    max2_WF: for (wf = 0; wf < CONV2_MAXPOOL_STEP; wf++) {
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

    uint16_t f;
    uint16_t w;
    uint16_t h;
    // For each feature
    flat_F: for (f = 0; f < CONV2_KERNELS; f++) {
    // Go through features rows and columns
        flat_H: for (h = 0; h < CONV2_MAXPOOL_FINAL_HEIGHT; h++){
            flat_W: for (w = 0; w < CONV2_MAXPOOL_FINAL_WIDTH; w++){
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
    uint16_t f;
    uint16_t nif;
    uint16_t nv;
    // Go through all the values of that feature
    fc1_F: for (f = 0; f < FC1_FEATURES; f++) {
        accumulated = 0;
        // For each feature
        fc1_NIF: for (nif = 0; nif < FC1_INPUT_FEATURES; nif++) {
            //int16_t op1 = add[ADD5]((int16_t)input_features[nif], (int16_t)-ZERO_CONV_2);
            //int16_t op2 = add[ADD6]((int16_t)input_weights[f][nif], (int16_t)-WEIGHTSQ_ZERO_FC1[f]);
            accumulated +=  mul[MUL3]((int16_t)input_features[nif], (int16_t)input_weights[f][nif]);
            //accumulated = (int)add[ADD3]((uint16_t)accumulated, (uint16_t)par_acc);
        }
        output_features[f] = requantize((accumulated + bias[f]), M_FC1[f], ZERO_FC1);
    }
}

void fullyConnected_2(const uint8_t input_features [FC1_FEATURES],
                      const int8_t input_weights [FC2_FEATURES][FC1_FEATURES],
                      const int32_t bias[FC2_FEATURES],
                      uint8_t output_features[FC2_FEATURES]){

    int32_t accumulated;
    uint16_t f;
    uint16_t nif;
    uint16_t nv;
    // Go through all the values of that feature
    fc2_F: for (f = 0; f < FC2_FEATURES; f++) {
        accumulated = 0;
        // For each feature
        fc2_NIF: for (nif = 0; nif < FC1_FEATURES; nif++) {
            int16_t op1 = add[ADD2]((int16_t)input_features[nif], (int16_t)-ZERO_FC1);
            //int16_t op2 = add[ADD8]((int16_t)input_weights[f][nif], (int16_t)-WEIGHTSQ_ZERO_FC2[f]);
            int16_t op2 = (int16_t)input_weights[f][nif];//, (int16_t)-WEIGHTSQ_ZERO_FC2[f]);
            accumulated += mul[MUL4](op1, op2);
            //accumulated = (int32_t)add[ADD4]((uint16_t)accumulated, (uint16_t)par_acc);
        }
        output_features[f] = requantize((accumulated + bias[f]), M_FC2[f], ZERO_FC2);
    }
}

void softMax(const uint8_t input_features[LAST_LAYER_FEATURES],
             const float input_features_scale,
             const int8_t input_features_zero,
             float output_features[LAST_LAYER_FEATURES]) {

    float input_features_float[LAST_LAYER_FEATURES];
    float accumulated = 0.0f;
    uint16_t f;
    // Dequantize to compute softmax
    sm_DQ: for (f = 0; f < LAST_LAYER_FEATURES; f++) {
        input_features_float[f] = input_features_scale * (input_features[f] - input_features_zero);
    }
    // Aggregate the exponential function of each feature in batches
    sm_EXP: for (f = 0; f < LAST_LAYER_FEATURES; f++) {
        accumulated += expf(input_features_float[f]);
    }
    // Divide the exponential function of each feature by the computed aggregation
    sm_F: for (f = 0; f < LAST_LAYER_FEATURES; f++) {
        output_features[f] = expf(input_features_float[f]) / accumulated;
    }
}

// Determine the classification provided by the CNN
int most_common_class(const float input_features[LAST_LAYER_FEATURES]) { // [features]

    int result = 0;
    uint16_t f;
    // Go through all features
    class_F: for (f = 1; f < LAST_LAYER_FEATURES; f++) {
        // Determine the position of the feature with the highest weight per batch
        if (input_features[f] > input_features[result]) {
            result = f;
        }
    }
    // Return this position
    return result;
}

int lenet_inference(const int8_t input_image[IMAGE_FEATURES][IMAGE_HEIGHT][IMAGE_WIDTH],
                    uint8_t output[LAST_LAYER_FEATURES]) {

    uint8_t padded_image[IMAGE_FEATURES][PADDED_IMAGE_HEIGHT][PADDED_IMAGE_WIDTH];
    uint8_t first_convolution[CONV1_KERNELS][CONV1_CONVOLVED_FEATURE_HEIGHT][CONV1_CONVOLVED_FEATURE_WIDTH];
    uint8_t first_convolution_relu[CONV1_KERNELS][CONV1_CONVOLVED_FEATURE_HEIGHT][CONV1_CONVOLVED_FEATURE_WIDTH];
    uint8_t first_convolution_max[CONV1_KERNELS][CONV1_MAXPOOL_FINAL_HEIGHT][CONV1_MAXPOOL_FINAL_WIDTH];

    uint8_t first_convolution_padded[CONV1_KERNELS][PADDED_FEATURES_WIDTH][PADDED_FEATURES_HEIGHT];
    uint8_t second_convolution[CONV2_KERNELS][CONV2_CONVOLVED_FEATURE_HEIGHT][CONV2_CONVOLVED_FEATURE_WIDTH];
    uint8_t second_convolution_relu[CONV2_KERNELS][CONV2_CONVOLVED_FEATURE_HEIGHT][CONV2_CONVOLVED_FEATURE_WIDTH];
    uint8_t second_convolution_max[CONV2_KERNELS][CONV2_MAXPOOL_FINAL_WIDTH][CONV2_MAXPOOL_FINAL_HEIGHT];

    uint8_t flatten[FC1_INPUT_FEATURES];
    uint8_t fc1[FC1_FEATURES];
    uint8_t last_layer[LAST_LAYER_FEATURES];

    padding_1(input_image, padded_image);
    convolution2DRelu_1(KERNELQ_CONV_1, BIAS_CONV1, padded_image, first_convolution_relu);
    maxPool_1(first_convolution_relu, first_convolution_max);

    padding_2(first_convolution_max, first_convolution_padded);
    convolution2DRelu_2(KERNELQ_CONV_2, BIAS_CONV2, first_convolution_padded, second_convolution_relu);
    maxPool_2(second_convolution_relu, second_convolution_max);

    flattenLayer(second_convolution_max, flatten);
    fullyConnected_1(flatten, WEIGHTSQ_FC1, BIAS_FC1, fc1);
    fullyConnected_2(fc1, WEIGHTSQ_FC2, BIAS_FC2, last_layer);

    for(int i=0; i< FC2_FEATURES; i++) output[i]=last_layer[i];
    // The position of the higest percentage is returned as the result
    float softmax[LAST_LAYER_FEATURES];
    softMax(output, SCALE_FC2, ZERO_FC2, softmax);

    int pred = most_common_class(softmax);

    return pred;
}

int main(int argc, char *argv[]) {
    MUL1 = atoi(argv[2]);
    MUL2 = atoi(argv[3]);
    MUL3 = atoi(argv[4]);
    MUL4 = atoi(argv[5]);

    ADD1 = atoi(argv[6]);
    ADD2 = atoi(argv[7]);
    /*ADD3 = 0;//atoi(argv[8]);
    ADD4 = 0;//atoi(argv[9]);
    ADD5 = 0;//atoi(argv[10]);
    ADD6 = 0;//atoi(argv[11]);
    ADD7 = 0;//atoi(argv[12]);
    ADD8 = 0;//atoi(argv[13]);*/

    //std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

    // Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
            mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    /*std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;*/

    int8_t sample_input[1][28][28]; // Read input from MiniLenetq_input_images.h
    uint8_t output[10]; // Output for 10 classes

    int error = 0;
    char* training = argv[1];
    int compare = strcmp(training,"training");
    
    vector<int> targets; //{0,0,1,1,2,2}
    vector<int> outputs;

    int tot;

    if (compare == 0)  {
        tot = dataset.training_images.size();
        //std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    } else {
        tot = dataset.test_images.size();
        //std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    }

    auto start = high_resolution_clock::now();

#pragma omp parallel for private(sample_input) shared(dataset)
    for (int img = 0; img < tot; img++) {
        int real;
        if (compare == 0)
            real = dataset.training_labels[img];
        else
            real = dataset.test_labels[img];
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                uint8_t pixel;
                if (compare == 0)
                    pixel = dataset.training_images[img][i * 28 + j];
                else
                    pixel = dataset.test_images[img][i * 28 + j];
                float pixelf = (((pixel / 255.0f) - 0.1307) / 0.3081);
                int8_t pixeli = (int8_t) roundf((pixelf / 0.0255) + 17);
                sample_input[0][i][j] = pixeli;
                //sample_input[0][i][j] = INPUT_IMAGES[img][0][i][j];
            }
        }

        //printf("image %d: %d\n", img, real);
        int pred = lenet_inference(sample_input, output);

        targets.push_back(real);
        outputs.push_back(pred);
        
        if (pred != real) {
            error++;
        }
    }
    
    auto stop = high_resolution_clock::now();
    std::chrono::duration<double> cpu_time = stop - start;

    float perc = (error / (float)tot);
    printf("%.4f", perc);
    /*printf("%10.4f s\n", cpu_time.count());*/
    
    string filename = training;
    string fullfilename1 = filename + "_confusion.log";
    
    Confusion confusion = Confusion(targets, outputs);
    ofstream CM(fullfilename1);//std::ios_base::app
    confusion.print(CM);

    string fullfilename2 = filename + "_evaluation.log";

    Evaluation evaluation = Evaluation(confusion);
    ofstream Eva(fullfilename2);//std::ios_base::app
    evaluation.print(Eva);//*/

    return 0;
}
//
// Created by tiago on 27/02/25.
//
