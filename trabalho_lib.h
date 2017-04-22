#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define FILENAME       "docword.nytimes.txt"
#define N_DOCS         3000
#define DOC_LINES      655468
#define DOC_COLS       3
#define N_VOCAB        102660

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