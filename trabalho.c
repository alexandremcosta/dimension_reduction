#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// #define FILENAME "docword.nytimes.small.txt" // only 10 documents
// #define N_DOCS    11
// #define DOC_LINES 2264
#define FILENAME "docword.nytimes.txt"
#define N_DOCS 3001
#define DOC_LINES 655468

#define DOC_COLS  3
#define N_VOCAB   102661

// Algorithms
double** create_bag();
double** create_distance(double**, int);
double   calculate_distance(double*, double*, int);
double** create_achlioptas(int);
double** create_gaussian(int);
double** create_projection(double**, double**, int);
double   max_distortion(double**, double**);

// Helpers
void     read_file(char[], size_t, size_t, size_t, int[][DOC_COLS]);
double** matrix_alloc(int, int);
void     free_matrix(double**, int);
void     check_memory(void *);
void     print_time(char[], clock_t, clock_t);
double   rand_normal(void);

int main()
{
    double** bag = create_bag();
    clock_t  begin = clock();
    double** distances = create_distance(bag, N_VOCAB);
    print_time("Distance calculation", begin, clock());
    srand(time(NULL));

    int dimensions[] = {4, 16, 64, 256, 1024, 4096, 15768};
    for (int i = 0; i < 5; i++)
    {
        int dim = dimensions[i];
        printf("Number of dimensions: %d\n", dim);

        // Achlioptas
        begin = clock();
        double** ach_matrix = create_achlioptas(dim);
        print_time("Achlioptas matrix", begin, clock());

        begin = clock();
        double** ach_projection = create_projection(bag, ach_matrix, dim);
        print_time("Achlioptas projection", begin, clock());

        begin = clock();
        double** ach_distances = create_distance(ach_projection, dim);
        print_time("Achlioptas distances", begin, clock());

        double ach_distortion = max_distortion(distances, ach_distances);
        printf("Achlioptas max distortion: %f\n", ach_distortion);

        printf("---\n");

        // Normal
        begin = clock();
        double** nor_matrix = create_gaussian(dim);
        print_time("Normal matrix", begin, clock());

        begin = clock();
        double** nor_projection = create_projection(bag, nor_matrix, dim);
        print_time("Normal projection", begin, clock());

        begin = clock();
        double** nor_distances = create_distance(nor_projection, dim);
        print_time("Normal distances", begin, clock());

        double nor_distortion = max_distortion(distances, nor_distances);
        printf("Normal max distortion: %f\n", nor_distortion);

        free_matrix(ach_matrix, N_VOCAB);
        free_matrix(ach_projection, N_DOCS);
        free_matrix(ach_distances, N_DOCS);
        free_matrix(nor_matrix, N_VOCAB);
        free_matrix(nor_projection, N_DOCS);
        free_matrix(nor_distances, N_DOCS);

        printf("############\n\n");
    }

    free_matrix(distances, N_DOCS);
    free_matrix(bag, N_DOCS);
    return 0;
}

double** create_bag()
{
    size_t docword_rows = DOC_LINES, docword_cols = DOC_COLS, docword_offset = 3; // small
    int docword[docword_rows][docword_cols];
    read_file(FILENAME, docword_offset, docword_rows, docword_cols, docword);

    double** bag = matrix_alloc(N_DOCS, N_VOCAB);
    for (int i = 0; i < N_DOCS; i++)
      for (int j = 0; j < N_VOCAB; j++)
        bag[i][j] = 0;

    for ( int i = 0; i < docword_rows; i++ )
        bag[docword[i][0]][docword[i][1]] = (int) docword[i][2];

    (bag[1][17654] == 4) ? printf("Bag created\n") : printf("Bag error\n");

    return bag;
}

double** create_achlioptas(int size)
{
    double** matrix = matrix_alloc(N_VOCAB, size);
    for (int i = 0; i < N_VOCAB; i++)
      for (int j = 0; j < size; j++)
      {
          double random = (double)rand() / (double)RAND_MAX;
          random = (random < (1./6) ? -1 : (random < (5./6) ? 0 : 1));
          random = random * sqrt(3./N_VOCAB);
          matrix[i][j] = random;
      }

    return matrix;
}

double** create_gaussian(int size)
{
    double** matrix = matrix_alloc(N_VOCAB, size);
    for (int i = 0; i < N_VOCAB; i++)
      for (int j = 0; j < size; j++)
      {
          matrix[i][j] = rand_normal();
      }

    return matrix;
}

double** create_distance(double** table, int n_cols)
{
    double** distances = matrix_alloc(N_DOCS, N_DOCS);
    for (int i = 1; i < N_DOCS; i++)
        for (int j = i+1; j < N_DOCS; j++)
            distances[i][j] = calculate_distance(table[i], table[j], n_cols);
    return distances;
}

double calculate_distance(double* a, double* b, int n_cols)
{
    double sum = 0;
    for ( int i = 0; i < n_cols; i++ )
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    return sqrt(sum);
}

double** create_projection(double** bag, double** matrix, int dim)
{
    double** result = matrix_alloc(N_DOCS, dim);

    for(int i=0; i < N_DOCS; i++)
        for(int j=0; j < dim; j++)
            result[i][j] = 0;

    for(int i = 0; i < N_DOCS; i++)
        for(int j = 0; j < dim; j++)
            for(int k = 0; k < N_VOCAB; k++)
                result[i][j] += bag[i][k] * matrix[k][j];

    return result;
}

double max_distortion(double** x, double** wx)
{
    double distortion, result = 0;

    for (int i = 1; i < N_DOCS; i++)
        for (int j = i+1; j < N_DOCS; j++)
        {
            distortion = fabs((wx[i][j] / x[i][j]) - 1);
            if (distortion > result)
                result = distortion;
        }

    return result;
}

// Helpers

void read_file(char filename[], size_t offset, size_t N, size_t M, int table[N][M])
{
    FILE *file = fopen(filename, "r");
    if ( file )
    {
        size_t i, j;

        for ( i = 0; i < N + offset; ++i )
        {
            if ( i < offset )
            {
                char line[256];
                fgets(line, sizeof line, file); // skip line
                continue;
            }
            for ( j = 0; j < M; ++j )
            {
                fscanf(file, "%d", &table[i-offset][j]);
            }
        }
        fclose(file);
    }
}

double** matrix_alloc(int rows, int cols)
{
  double** space = malloc(rows * sizeof(double*));
  check_memory(space);
  for (int i = 0; i < rows; i++)
  {
    space[i] = malloc(cols * sizeof(double));
    check_memory(space[i]);
  }
  return space;
}

void free_matrix(double** matrix, int rows)
{
  for(int i = 0; i < rows; i++)
    free(matrix[i]);
  free(matrix);
}

void check_memory(void* variable)
{
  if (variable == NULL)
  {
    printf("Memory allocation error!");
    exit(0);
  }
}

void print_time(char msg[], clock_t begin, clock_t end)
{
    double elapsed = (double)(end - begin) / CLOCKS_PER_SEC;

    printf(msg);
    printf(": %f seconds\n", elapsed);
}

// https://phoxis.org/2013/05/04/generating-random-numbers-from-normal-distribution-in-c/
double rand_normal(void)
{
    double mu = 0, sigma = 1.0/N_VOCAB;
    double U1, U2, W, mult;
    static double X1, X2;
    static int call = 0;

    if (call == 1)
    {
        call = !call;
        return (mu + sigma * (double) X2);
    }

    do
    {
        U1 = -1 + ((double) rand () / RAND_MAX) * 2;
        U2 = -1 + ((double) rand () / RAND_MAX) * 2;
        W = pow (U1, 2) + pow (U2, 2);
    }
    while (W >= 1 || W == 0);

    mult = sqrt ((-2 * log (W)) / W);
    X1 = U1 * mult;
    X2 = U2 * mult;

    call = !call;

    return (mu + sigma * (double) X1);
}


        // double progress = 100.0*i/N_DOCS;
        // double elapsed = (double)(clock() - begin) / CLOCKS_PER_SEC;
        // double total = elapsed*N_DOCS/i;
        // printf("%.5f%% - Elapsed time: %.4f minutes - Remaining time: %.4f minutes \r", progress, elapsed/60, (total - elapsed)/60); fflush(stdout);

