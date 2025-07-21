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
#define COMMENT "highpass filter"
#define RGB_COMPONENT_COLOR 255

void highpass(unsigned char* in, unsigned char* out, int HEIGHT, int WIDTH, int mul1, int add1, int add2, int add3, int add4) {

    loop1: for (int i = 0; i < (HEIGHT/8)*8; i++) {
        loop2: for (int j = 0; j < (WIDTH/8)*8; j++) {
            // Set output to 0 if the 3x3 receptive field is out of bound.
            if ((i < 1) | (i > HEIGHT - 2) | (j < 1) | (j > WIDTH - 2)) {
                out[i * HEIGHT + j] = 0;
                continue;
            }
            int x1, x2, x3, x4, sum;

            x1 = mul[mul1](in[i * HEIGHT + j], 5);
            x2 = (int)-add[add1]((int)-x1, in[(i - 1) * HEIGHT + j]);
            x3 = (int)-add[add2]((int)-x2, in[i * HEIGHT + (j - 1)]);
            x4 = (int)-add[add3]((int)-x3, in[i * HEIGHT + (j + 1)]);
            sum = (int)-add[add4]((int)-x4, in[(i + 1) * HEIGHT + j]);

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
    strcat(outname, ".high.ppm");

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

        highpass(inputImage, outImage, image->y, image->x, atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), atoi(argv[7]));

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

