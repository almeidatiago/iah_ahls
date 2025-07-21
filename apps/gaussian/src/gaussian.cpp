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

//#define HEIGHT 512
//#define WIDTH 512
#define COMMENT "gaussian filter"
#define RGB_COMPONENT_COLOR 255

void gaussian(unsigned char* in, unsigned char* out, int HEIGHT, int WIDTH, int mul1, int mul2, int mul3, int mul4, int add1, int add2, int add3, int add4) {

    loop1: for (int i = 0; i < (HEIGHT/8)*8; i++) {
        loop2: for (int j = 0; j < (WIDTH/8)*8; j++) {
        // Set output to 0 if the 3x3 receptive field is out of bound.
            if ((i < 3) | (i > HEIGHT - 4) | (j < 3) | (j > WIDTH - 4)) {
                out[i * HEIGHT + j] = 0;
                continue;
            }
            int x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20,
                    x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40,
                    x41, x42, x43, x44, x45, x46, x47, x48, x49, x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, sum;

            x1 = mul[mul1](in[i * HEIGHT + j], 16);
            x2 = mul[mul2](in[(i - 1) * HEIGHT + (j - 1)], 5);
            x3 = mul[mul2](in[(i - 1) * HEIGHT + j], 5);
            x4 = mul[mul2](in[(i - 1) * HEIGHT + (j + 1)], 5);
            x5 = mul[mul2](in[i * HEIGHT + (j - 1)], 5);
            x6 = mul[mul2](in[i * HEIGHT + (j + 1)], 5);
            x7 = mul[mul2](in[(i + 1) * HEIGHT + (j - 1)], 5);
            x8 = mul[mul2](in[(i + 1) * HEIGHT + j], 5);
            x9 = mul[mul2](in[(i + 1) * HEIGHT + (j + 1)], 5);
            x10 = mul[mul3](in[(i - 2) * HEIGHT + (j - 1)], -3);
            x11 = mul[mul3](in[(i - 2) * HEIGHT + j], -3);
            x12 = mul[mul3](in[(i - 2) * HEIGHT + (j + 1)], -3);
            x13 = mul[mul3](in[(i - 1) * HEIGHT + (j - 2)], -3);
            x14 = mul[mul3](in[(i - 1) * HEIGHT + (j + 2)], -3);
            x15 = mul[mul3](in[i * HEIGHT + (j - 2)], -3);
            x16 = mul[mul3](in[i * HEIGHT + (j + 2)], -3);
            x17 = mul[mul3](in[(i + 1) * HEIGHT + (j - 2)], -3);
            x18 = mul[mul3](in[(i + 1) * HEIGHT + (j + 2)], -3);
            x19 = mul[mul3](in[(i + 2) * HEIGHT + (j - 1)], -3);
            x20 = mul[mul3](in[(i + 2) * HEIGHT + j], -3);
            x21 = mul[mul3](in[(i + 2) * HEIGHT + (j + 1)], -3);
            x22 = mul[mul4](in[(i - 2) * HEIGHT + (j - 2)], -2);
            x23 = mul[mul4](in[(i - 2) * HEIGHT + (j + 2)], -2);
            x24 = mul[mul4](in[(i + 2) * HEIGHT + (j - 2)], -2);
            x25 = mul[mul4](in[(i + 2) * HEIGHT + (j + 2)], -2);

            x26 = add[add1](x1, x2);
            x27 = add[add1](x3, x4);
            x28 = add[add1](x5, x6);
            x29 = add[add1](x7, x8);
            x30 = add[add1](x9, x10);
            x31 = add[add2](x11, x12);
            x32 = add[add2](x13, x14);
            x33 = add[add2](x15, x16);
            x34 = add[add2](x17, x18);
            x35 = add[add2](x19, x20);
            x36 = add[add2](x21, x22);
            x37 = add[add3](x23, x24);
            x38 = add[add3](x25, x26);
            x39 = add[add4](x27, x28);
            x40 = add[add4](x29, x30);
            x41 = add[add4](x31, x32);
            x42 = add[add4](x33, x34);
            x43 = add[add4](x35, x36);
            x44 = add[add4](x37, x38);
            x45 = add[add4](x39, x40);
            x46 = add[add4](x41, x42);
            x47 = add[add4](x43, x44);
            x48 = add[add4](x45, x46);
            x49 = add[add4](x47, x48);

            x50 = x49 - in[(i - 3) * HEIGHT + (j - 1)];
            x51 = x50 - in[(i - 3) * HEIGHT + j];
            x52 = x51 - in[(i - 3) * HEIGHT + (j + 1)];
            x53 = x52 - in[(i - 1) * HEIGHT + (j - 3)];
            x54 = x53 - in[(i - 1) * HEIGHT + (j + 3)];
            x55 = x54 - in[i * HEIGHT + (j - 3)];
            x56 = x55 - in[i * HEIGHT + (j + 3)];
            x57 = x56 - in[(i + 1) * HEIGHT + (j - 3)];
            x58 = x57 - in[(i + 1) * HEIGHT + (j + 3)];
            x59 = x58 - in[(i + 3) * HEIGHT + (j - 1)];
            x60 = x59 - in[(i + 3) * HEIGHT + j];
            sum = x60 - in[(i + 3) * HEIGHT + (j + 1)];

            sum = (sum < 0) ? 0 : sum;
            sum = (sum > 255) ? 255 : sum;
            out[i * HEIGHT + j] = (unsigned char)sum;
        }
    }
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
        fprintf(stderr, "Unable to open file '%s'\n", filename);
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
    strcat(outname, ".gauss.ppm");

    if ((output = fopen(outname, "w")) == NULL) {
        fprintf(stderr, "Error: could not open output file!\n");
        exit(1);
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
    char filename[30];
    unsigned size;
    char * kind;
    
    kind = argv[2];

    if ((input = fopen(argv[1], "r")) == NULL) {
        fprintf(stderr, "Error: could not open file\n");
        return 1;
    }

    // Read input filename
    fscanf(input, "%u", &size);
    
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

        gaussian(inputImage, outImage, image->y, image->x, atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), atoi(argv[7]), atoi(argv[8]), atoi(argv[9]), atoi(argv[10]));

        writePPM(filesList[i], outImage, inputList[i], kind, image->y, image->x);
        
        //printf("%s -> %s\n", inputList[i], filesList[i]);

        free(image);
        free(inputImage);
        free(outImage);
    }
    t = omp_get_wtime() - t;
    //time_taken = ((double) t) / CLOCKS_PER_SEC; // in seconds
    
    print(filesList, size);
    
    return EXIT_SUCCESS;
    
}

