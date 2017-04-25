#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N_DOCS    3000
#define DOC_LINES 655468
#define FILENAME  "docword.nytimes.txt"

// Change number of documents to run faster
// #define N_DOCS    10
// #define DOC_LINES 2264
//
// #define N_DOCS    100
// #define DOC_LINES 21853
//
// #define N_DOCS    200
// #define DOC_LINES 44412
//
// #define N_DOCS    400
// #define DOC_LINES 91620
//
// Smaller docword to be used with 400 docs maximum
// #define FILENAME  "docword.nytimes.small.txt"

#define DOC_COLS  3
#define N_VOCAB   102660

float** create_bag();
float** create_distance(float**, int);
float   calculate_distance(float*, float*, int);
float   max_distortion(float**, float**);
float   avg_distortion(float**, float**);

void    read_file(char[], size_t, size_t, size_t, int[][DOC_COLS]);
void    print_time(clock_t, clock_t);
float** load_distance();
float** mat_alloc(int, int);
void    free_matrix(float**, int);
void    check_memory(void *);
float** mat_mult(int, int, float *const *, int, float *const *);
float** mat_transpose(int, int, float *const*);
