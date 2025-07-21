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
#define COMMENT "smooth filter"
#define RGB_COMPONENT_COLOR 255

void smooth(unsigned char* in, unsigned char* out, int HEIGHT, int WIDTH, int add1, int add2, int add3, int add4, int add5, int add6) {

    loop1: for (int i = 0; i < (HEIGHT/8)*8; i++) {
        loop2: for (int j = 0; j < (WIDTH/8)*8; j++) {
        // Set output to 0 if the 3x3 receptive field is out of bound.
        if ((i < 2) | (i > HEIGHT - 3) | (j < 2) | (j > WIDTH - 3)) {
            out[i * HEIGHT + j] = 0;
            continue;
        }
            int x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20,
                    x21, x22, x23, sum;

            x1 = add[add1](in[(i - 2) * HEIGHT + (j - 2)], in[(i - 2) * HEIGHT + (j - 1)]);
            x2 = add[add1](in[(i - 2) * HEIGHT + (j - 0)], in[(i - 2) * HEIGHT + (j + 1)]);
            x3 = add[add1](x1, in[(i - 2) * HEIGHT + (j + 2)]);
            x4 = add[add1](x2, x3);

            x5 = add[add2](in[(i - 1) * HEIGHT + (j - 2)], in[(i - 1) * HEIGHT + (j - 1)]);
            x6 = add[add2](in[(i - 1) * HEIGHT + (j - 0)], in[(i - 1) * HEIGHT + (j + 1)]);
            x7 = add[add2](x5, in[(i - 1) * HEIGHT + (j + 2)]);
            x8 = add[add2](x6, x7);

            x9 = add[add3](in[(i - 0) * HEIGHT + (j - 2)], in[(i - 0) * HEIGHT + (j - 1)]);
            x10 = add[add3](in[(i - 0) * HEIGHT + (j - 0)], in[(i - 0) * HEIGHT + (j + 1)]);
            x11 = add[add3](x9, in[(i - 0) * HEIGHT + (j + 2)]);
            x12 = add[add3](x10, x11);

            x13 = add[add4](in[(i + 1) * HEIGHT + (j - 2)], in[(i + 1) * HEIGHT + (j - 1)]);
            x14 = add[add4](in[(i + 1) * HEIGHT + (j - 0)], in[(i + 1) * HEIGHT + (j + 1)]);
            x15 = add[add4](x13, in[(i + 1) * HEIGHT + (j + 2)]);
            x16 = add[add4](x14, x15);

            x17 = add[add5](in[(i + 2) * HEIGHT + (j - 2)], in[(i + 2) * HEIGHT + (j - 1)]);
            x18 = add[add5](in[(i + 2) * HEIGHT + (j - 0)], in[(i + 2) * HEIGHT + (j + 1)]);
            x19 = add[add5](x17, in[(i + 2) * HEIGHT + (j + 2)]);
            x20 = add[add5](x18, x19);

            x21 = add[add6](x4, x8);
            x22 = add[add6](x12, x16);
            x23 = add[add6](x21, x22);
            sum = add[add6](x23, x20);

            sum = sum / 25;

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
    strcat(outname, ".smoo.ppm");

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
    
//#pragma omp parallel for shared(inputList, filesList)
    for (int i = 0; i < size; i++) {
        
        PPMImage *image = readPPM(inputList[i]);
        unsigned char *inputImage = (unsigned char *) malloc(sizeof(unsigned char) * image->y * image->x);
        unsigned char *outImage = (unsigned char *) malloc(sizeof(unsigned char) * image->y * image->x);

        init(inputImage, image);

        smooth(inputImage, outImage, image->y, image->x, atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), atoi(argv[7]), atoi(argv[8]));

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
