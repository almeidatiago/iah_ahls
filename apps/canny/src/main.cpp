#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#include "components.h"
#include <cmath>
#include <iostream>

#include <stdlib.h>
#include <string.h>
#include <cstdint>

#define HEIGHT 512
#define WIDTH 512
#define COMMENT "Canny filter"
#define RGB_COMPONENT_COLOR 255

            // gradient hypot & direction
 #define SIZE HEIGHT * WIDTH

void Canny(unsigned char* dstimg, unsigned char* srcimg, int width, int height,
              int mul1, int mul2, int mul3, int mul4, int mul5, int mul6, int mul7, int mul8, int mul9,
              int add1, int add2, int add3, int add4, int add5, int add6, int add7, int add8, int add9, int add10, int add11, int add12, int add13, int add14, int add15, int add16
) {

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

    for (int xy = 0; xy < SIZE; xy++)
        src[xy] = srcimg[xy];
    
    //std::cout << src[0] << std::endl;

    // gaussian filter
    for (int x = 0; x < w_; x++) {
        for (int y = 0; y < h_; y++) {
            int pos = x + (y * w_);
            if (x < offset_xy || x >= (w_ - offset_xy) || y < offset_xy ||
                y >= (h_ - offset_xy)) {
                dst[pos] = src[pos];
            } else {
                int x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, sum;

                x1 = add[add1](src[(x - 1) + ((y - 1) * w_)], src[(x + 1) + ((y - 1) * w_)]);
                x2 = mul[mul1](src[x + ((y - 1) * w_)], 2);
                x3 = add[add2](x1, x2);
                x4 = mul[mul2](src[(x - 1) + (y * w_)], 2);
                x5 = mul[mul3](src[x + (y * w_)], 4);
                x6 = mul[mul4](src[(x + 1) + (y * w_)], 2);
                x7 = add[add3](x4, x5);
                x8 = add[add4](x6, x7);
                x9 = add[add5](src[(x - 1) + ((y + 1) * w_)], src[(x + 1) + ((y + 1) * w_)]);
                x10 = mul[mul5](src[x + ((y + 1) * w_)], 2);
                x11 = add[add6](x9, x10);
                x12 = add[add7](x3, x8);
                sum = add[add8](x11, x12);

                sum /= 16;
                sum = (sum < 0) ? 0 : sum;
                sum = (sum > 255) ? 255 : sum;

                dst[pos] = (unsigned char) sum;
            }
        }
    }

    // apply sobel kernels
    for (int x = offset_xy; x < w_ - offset_xy; x++) {
        for (int y = offset_xy; y < h_ - offset_xy; y++) {
            int src_pos = x + (y * w_);
            int sobel, theta;
            int x1, x2, x3, x4, x5, x6;
            int gx_sum, gy_sum;

            x1 = add[add9]((int16_t)dst[(y - 1) * w_ + (x - 1)], (int16_t)dst[(y + 1) * w_ + (x - 1)]);
            x2 = mul[mul6]((int16_t)dst[y * w_ + (x - 1)], (int16_t)2);
            x3 = add[add10]((int16_t)x1, (int16_t)x2);
            x4 = add[add11]((int16_t)dst[(y - 1) * w_ + (x + 1)], (int16_t)dst[(y + 1) * w_ + (x + 1)]);
            x5 = mul[mul7]((int16_t)dst[y * w_ + (x + 1)], (int16_t)2);
            x6 = add[add12]((int16_t)x4, (int16_t)x5);
            gx_sum = x3 - x6;
            gx_sum = (gx_sum < 0) ? -gx_sum : gx_sum;

            x1 = add[add13]((int16_t)dst[(y - 1) * w_ + (x - 1)], (int16_t)dst[(y - 1) * w_ + (x + 1)]);
            x2 = mul[mul8]((int16_t)dst[(y - 1) * w_ + x], (int16_t)2);
            x3 = add[add14]((int16_t)x1, (int16_t)x2);
            x4 = add[add15]((int16_t)dst[(y + 1) * w_ + (x - 1)], (int16_t)dst[(y + 1) * w_ + (x + 1)]);
            x5 = mul[mul9]((int16_t)dst[(y + 1) * w_ + x], (int16_t)2);
            x6 = add[add16]((int16_t)x4, (int16_t)x5);
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
    for (int x = offset_xy; x < w_ - offset_xy; x++) {
        for (int y = offset_xy; y < h_ - offset_xy; y++) {
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
    for (int x = 1; x < w_ - 1; x++) {
        for (int y = 1; y < h_ - 1; y++) {
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
    for (int x = 0; x < w_; x++) {
        for (int y = 0; y < h_; y++) {
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
    for (int x = 1; x < w_ - 1; x++) {
        for (int y = 1; y < h_ - 1; y++) {
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

    for (int xy = 0; xy < SIZE; xy++)
        dstimg[xy] = dst_h[xy];
}

typedef struct {
    unsigned char red, green, blue;
} PPMPixel;

typedef struct {
    int x, y;
    PPMPixel *data;
} PPMImage;

static PPMImage *readPPM(const char *filename) {
    char buff[16];
    PPMImage *img;
    FILE *fp;
    int c, rgb_comp_color;
    fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "PPMImage: Unable to open file '%s'\n", filename);
        exit(1);
    }

    if (!fgets(buff, sizeof(buff), fp)) {
        perror(filename);
        exit(1);
    }

    if (buff[0] != 'P' || buff[1] != '6') {
        fprintf(stderr, "Invalid image format (must be 'P6')\n");
        exit(1);
    }

    img = (PPMImage *)malloc(sizeof(PPMImage));
    if (!img) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    c = getc(fp);
    while (c == '#') {
        while (getc(fp) != '\n')
            ;
        c = getc(fp);
    }

    ungetc(c, fp);
    if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
        fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
        exit(1);
    }

    if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
        fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
        exit(1);
    }

    if (rgb_comp_color != RGB_COMPONENT_COLOR) {
        fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
        exit(1);
    }

    while (fgetc(fp) != '\n')
        ;
    img->data = (PPMPixel *)malloc(img->x * img->y * sizeof(PPMPixel *));

    if (!img) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
        fprintf(stderr, "Error loading image '%s'\n", filename);
        exit(1);
    }

    fclose(fp);
    return img;
}

void writePPM(char* outname, unsigned char* img, char *filename, const char * kind, int height, int width) {
    FILE *output;
    
    char *token = strtok(filename, "p");
    strcpy(outname, token);

    strcat(outname, kind);
    strcat(outname, ".canny.ppm");

    if ((output = fopen(outname, "w")) == NULL) {
        fprintf(stderr, "Error: could not open output file!\n");
    }

    fprintf(output, "P2\n");
    fprintf(output, "%d %d\n", height, width);
    fprintf(output, "%d\n", RGB_COMPONENT_COLOR);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++)
            fprintf(output, "%d ", img[i * width + j]);
        fprintf(output, "\n");
    }
    fclose(output);
}

void init(unsigned char* buf, PPMImage * image) {
	for (int i = 0; i < image->y * image->x; i++) {
		buf[i] = image->data[i].red;
	}
}

// Output to stdout
void print(char **r, unsigned size) {
    for (int i = 0; i < size; i++) {
        printf(" %s", r[i]);
    }
    //printf("\n");
}

int main(int argc, char *argv[]) {
    FILE *input;
    char filename[255];
    unsigned size;
    char * kind;
    
    kind = argv[2];

    if ((input = fopen(argv[1], "r")) == NULL) {
        fprintf(stderr, "Error: could not open file\n");
        return 1;
    }

    // Read input filename
    fscanf(input, "%u", &size);
    
//    omp_set_num_threads(8);
    
    char *filesList[size];
    char *inputList[size];
    
    for (int i = 0; i < size; i++) {
        fscanf(input, "%s\n", filename);
        inputList[i] = (char*) malloc (strlen(filename)+1);
        filesList[i] = (char*) malloc (strlen(filename)+8);
        strcpy(inputList[i],filename);
    }

    double t;
    double time_taken;
    t = omp_get_wtime();
    
#pragma omp parallel for shared(inputList, filesList)
    for (int i = 0; i < size; i++) {
        
        PPMImage *image = readPPM(inputList[i]);
        unsigned char *inputImage = (unsigned char *) malloc(sizeof(unsigned char) * image->y * image->x);
        unsigned char *outImage = (unsigned char *) malloc(sizeof(unsigned char) * image->y * image->x);

        init(inputImage, image);

        Canny(outImage, inputImage, image->x, image->y, 
              atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), atoi(argv[7]), atoi(argv[8]), atoi(argv[9]), atoi(argv[10]), atoi(argv[11]), 
              atoi(argv[12]), atoi(argv[13]), atoi(argv[14]), atoi(argv[15]), atoi(argv[16]), atoi(argv[17]), atoi(argv[18]), atoi(argv[19]), atoi(argv[20]), atoi(argv[21]), atoi(argv[22]), atoi(argv[23]), atoi(argv[24]), atoi(argv[25]), atoi(argv[26]), atoi(argv[27])
             );

        writePPM(filesList[i], outImage, inputList[i], kind, image->y, image->x);
        
        //printf("%s -> %s\n", inputList[i], filesList[i]);

        free(image);
        free(inputImage);
        free(outImage);
    }
    t = omp_get_wtime() - t;
    //time_taken = ((double) t) / CLOCKS_PER_SEC; // in seconds
    
    print(filesList, size);
    
    //printf("\n\nTook %f seconds to execute \n\n", t);
    
    return EXIT_SUCCESS;
}

