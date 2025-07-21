/*
 * Copyright (c) 2021-2023 Universitat Politècnica de València
 * Authors: David de Andrés and Juan Carlos Ruiz
 *          Fault-Tolerant Systems
 *          Instituto ITACA
 *          Universitat Politècnica de València
 *
 * Distributed under MIT License
 * (See accompanying file LICENSE.txt)
 */

#ifndef CNN_DEFINES

#define CNN_DEFINES

// Number of images in the .h file used for testing
#define NUM_IMAGES 200
// Number of images in the plain text file used for testing and inference
#define NUM_IMAGES_FROM_FILE 10000

// Parameters of the input images
#define IMAGE_FEATURES 1
#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28

// Number of categories
#define LAST_LAYER_FEATURES 10

// Just for debugging
#define PRINT 1
#define DO_NOT_PRINT 0

// Defines related to the first padding
#define PADDING 2
#define PADDED_IMAGE_WIDTH IMAGE_WIDTH + (2 * PADDING)
#define PADDED_IMAGE_HEIGHT IMAGE_HEIGHT + (2 * PADDING)

// Defines related to the first 2D convolution
#define CONV1_FEATURES IMAGE_FEATURES
#define CONV1_FEATURE_HEIGHT PADDED_IMAGE_HEIGHT
#define CONV1_FEATURE_WIDTH PADDED_IMAGE_WIDTH
#define CONV1_KERNELS 3
#define CONV1_KERNEL_WIDTH 5
#define CONV1_KERNEL_HEIGHT 5
#define CONV1_CONVOLVED_FEATURE_HEIGHT PADDED_IMAGE_HEIGHT - CONV1_KERNEL_HEIGHT + 1
#define CONV1_CONVOLVED_FEATURE_WIDTH PADDED_IMAGE_WIDTH - CONV1_KERNEL_WIDTH + 1

// Defines related to the first max pooling
#define CONV1_MAXPOOL_STEP 2
#define CONV1_MAXPOOL_FINAL_WIDTH (CONV1_CONVOLVED_FEATURE_HEIGHT) / (CONV1_MAXPOOL_STEP)
#define CONV1_MAXPOOL_FINAL_HEIGHT (CONV1_CONVOLVED_FEATURE_WIDTH) / (CONV1_MAXPOOL_STEP)

// Defines related to the second padding
#define PADDED_FEATURES_WIDTH CONV1_MAXPOOL_FINAL_WIDTH + (2 * PADDING)
#define PADDED_FEATURES_HEIGHT CONV1_MAXPOOL_FINAL_HEIGHT + (2 * PADDING)

// Defines related to the second 2D convolution
#define CONV2_KERNELS 6
#define CONV2_KERNEL_WIDTH 5
#define CONV2_KERNEL_HEIGHT 5
#define CONV2_CONVOLVED_FEATURE_WIDTH PADDED_FEATURES_WIDTH - CONV2_KERNEL_WIDTH + 1
#define CONV2_CONVOLVED_FEATURE_HEIGHT PADDED_FEATURES_HEIGHT - CONV2_KERNEL_HEIGHT + 1

// Defines related to the second max pooling
#define CONV2_MAXPOOL_STEP 2
#define CONV2_MAXPOOL_FINAL_WIDTH (CONV2_CONVOLVED_FEATURE_WIDTH) / (CONV2_MAXPOOL_STEP)
#define CONV2_MAXPOOL_FINAL_HEIGHT (CONV2_CONVOLVED_FEATURE_HEIGHT) / (CONV2_MAXPOOL_STEP)

// Defines related to the first and second fully connected layers
#define FC1_INPUT_FEATURES CONV2_KERNELS * CONV2_MAXPOOL_FINAL_WIDTH * CONV2_MAXPOOL_FINAL_HEIGHT
#define FC1_FEATURES (FC1_INPUT_FEATURES) / 2
#define FC2_FEATURES 10
 
#endif
