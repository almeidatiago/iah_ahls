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
 

#ifndef CLASSIFICATION_RES

#define CLASSIFICATION_RES

// Expected classification for the input images
static const int res_cnn_prediction[NUM_IMAGES] =
{9,2,0,3,4,0,0,4,7,6,1,1,3,9,9,7,9,7,1,4,9,2,8,8,8,4,9,9,5,9,8,1,1,1,0,3,2,4,2,6,9,1,6,5,4,9,4,1,7,8,2,6,1,9,6,0,0,4,1,6,6,8,2,5,1,4,8,6,0,6,2,0,7,3,4,6,7,1,7,6,7,1,0,4,5,9,6,0,0,9,5,1,1,2,9,2,4,8,2,2,1,2,4,6,7,4,3,3,4,9,1,8,8,3,6,0,5,0,4,1,4,0,3,3,1,7,1,6,0,1,2,9,6,7,5,5,6,2,3,8,9,2,1,9,0,3,9,4,6,1,3,2,8,9,4,0,9,6,2,9,8,1,4,0,4,5,5,1,6,6,9,1,8,1,2,9,7,9,3,6,9,9,0,1,4,9,3,7,3,3,5,2,8,3,7,7,7,4,1,7};

// Actual classification for the input images
static const int res_cnn_real[NUM_IMAGES] =
{9,2,0,3,4,0,0,4,7,6,1,1,3,9,9,7,9,7,1,4,9,2,8,8,8,4,9,9,5,9,8,1,1,1,0,3,2,4,2,6,9,1,6,5,4,9,4,1,7,8,2,6,1,9,6,0,0,4,1,6,6,8,2,5,1,4,8,6,0,6,2,0,7,3,4,6,7,1,7,6,7,1,0,4,5,9,6,0,0,9,5,1,1,2,9,2,4,8,2,2,1,2,4,6,7,4,3,3,4,9,1,8,8,3,6,0,5,0,4,1,4,0,3,3,1,7,1,6,0,1,2,9,6,7,5,5,6,2,3,8,9,2,1,9,0,3,9,4,6,1,3,2,8,9,4,0,9,6,2,9,8,1,4,0,4,5,5,1,6,6,9,1,8,1,2,9,7,9,3,6,9,9,0,1,4,9,3,7,3,3,5,2,8,3,7,7,7,4,1,7};

#endif
