#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// #define FILENAME "docword.nytimes.small.txt" // only 10 documents
// #define N_DOCS    10
// #define DOC_LINES 2264
#define FILENAME "docword.nytimes.txt"
#define N_DOCS 3000
#define DOC_LINES 655468

#define DOC_COLS  3
#define N_VOCAB   102660

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
void     print_time(char[], clock_t, clock_t);
double** load_distance(double**);
void     save_distance(double** table);
double   rand_normal(void);
double** mat_alloc(int, int);
void     free_matrix(double**, int);
void     check_memory(void *);
double** mat_mul1(int, int, double *const *, int, double *const *);
double** mat_mul2(int, int, double *const *, int, double *const *);

int main()
{
    clock_t  begin;
    double** bag = create_bag();
    double** distances = load_distance(bag);

    srand(time(NULL));

    int dimensions[] = {4, 16, 64, 256, 1024, 4096, 15768};
    for (int i = 0; i < 6; i++)
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

    double** bag = mat_alloc(N_DOCS, N_VOCAB);
    for (int i = 0; i < N_DOCS; i++)
      for (int j = 0; j < N_VOCAB; j++)
        bag[i][j] = 0;

    for ( int i = 0; i < docword_rows; i++ )
        bag[docword[i][0]-1][docword[i][1]-1] = (int) docword[i][2];

    (bag[0][17653] == 4) ? printf("Bag created\n") : printf("Bag error\n");

    return bag;
}

double** create_achlioptas(int size)
{
    double** matrix = mat_alloc(N_VOCAB, size);
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
    double** matrix = mat_alloc(N_VOCAB, size);
    for (int i = 0; i < N_VOCAB; i++)
      for (int j = 0; j < size; j++)
      {
          matrix[i][j] = rand_normal();
      }

    return matrix;
}

double** create_distance(double** table, int n_cols)
{
    double** distances = mat_alloc(N_DOCS, N_DOCS);
    double distance;

    for (int i = 0; i < N_DOCS; i++)
        for (int j = i+1; j < N_DOCS; j++)
        {
            distance = calculate_distance(table[i], table[j], n_cols);
            distances[i][j] = distance;
        }

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
    return mat_mul2(N_DOCS, N_VOCAB, bag, dim, matrix);
}

double max_distortion(double** x, double** wx)
{
    double distortion, result = 0;

    for (int i = 0; i < N_DOCS; i++)
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

double** load_distance(double** bag)
{
    double** table;

    // if (N_DOCS == 10)
    // {
        clock_t begin = clock();
        table = create_distance(bag, N_VOCAB);
        print_time("Distance calculation", begin, clock());
        save_distance(table);
    // }
    // else {
    //     FILE *file = fopen("distances.txt", "r");
    //     table = mat_alloc(N_DOCS, N_DOCS);
    //     if ( file )
    //     {
    //         size_t i, j;
    //         for ( i = 0; i < N_DOCS; ++i )
    //             for ( j = 0; j < N_DOCS; ++j )
    //                 fscanf(file, "%lf ", &table[i][j]);
    //         fclose(file);
    //     }
    // }
    return table;
}

void save_distance(double** table)
{
    FILE *file = fopen("distances.txt", "w+");
    if ( file )
    {
        size_t i, j;

        for ( i = 0; i < N_DOCS; ++i )
            for ( j = 0; j < N_DOCS; ++j )
                fprintf(file, "%a ", table[i][j]);
        fclose(file);
    }
}

double** mat_alloc(int rows, int cols)
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

double** mat_transpose(int rows, int cols, double *const* a)
{
	double** m;
	m = mat_alloc(cols, rows);

	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			m[j][i] = a[i][j];

	return m;
}

double** mat_mul1(int n_a_rows, int n_a_cols, double *const *a, int n_b_cols, double *const *b)
{
	int i, j, k, n_b_rows = n_a_cols;
	double **matrix, **bT;
	matrix = mat_alloc(n_a_rows, n_b_cols);
	bT = mat_transpose(n_b_rows, n_b_cols, b);

	for (i = 0; i < n_a_rows; i++) {
		const double *ai = a[i];
		double *mi = matrix[i];

		for (j = 0; j < n_b_cols; j++) {
			double t = 0.0f, *bTj = bT[j];

			for (k = 0; k < n_a_cols; k++)
				t += ai[k] * bTj[k];
			mi[j] = t;
		}
	}
	free_matrix(bT, n_b_cols);

	return matrix;
}

#include <emmintrin.h>
double sdot_sse(int n, const double *x, const double *y)
{
    int i, n8 = n>>3<<3;
    __m128d vs1, vs2;
    double s, t[4];
    vs1 = _mm_setzero_pd();
    vs2 = _mm_setzero_pd();
    for (i = 0; i < n8; i += 8) {
        __m128d vx1, vx2, vy1, vy2;
        vx1 = _mm_loadu_pd(&x[i]);
        vx2 = _mm_loadu_pd(&x[i+4]);
        vy1 = _mm_loadu_pd(&y[i]);
        vy2 = _mm_loadu_pd(&y[i+4]);
        vs1 = _mm_add_pd(vs1, _mm_mul_pd(vx1, vy1));
        vs2 = _mm_add_pd(vs2, _mm_mul_pd(vx2, vy2));
    }
    for (s = 0.0f; i < n; ++i) s += x[i] * y[i];
    _mm_storeu_pd(t, vs1);
    s += t[0] + t[1] + t[2] + t[3];
    _mm_storeu_pd(t, vs2);
    s += t[0] + t[1] + t[2] + t[3];
    return s;
}

double **mat_mul2(int n_a_rows, int n_a_cols, double *const *a, int n_b_cols, double *const *b)
{
    int i, j, ii, jj, x = 16, n_b_rows = n_a_cols;
    double **m, **bT;
    m = mat_alloc(n_a_rows, n_b_cols);
    bT = mat_transpose(n_b_rows, n_b_cols, b);
    for (i = 0; i < n_a_rows; i += x) {
        for (j = 0; j < n_b_cols; j += x) {
            int je = n_b_cols < j + x? n_b_cols : j + x;
            int ie = n_a_rows < i + x? n_a_rows : i + x;
            for (ii = i; ii < ie; ++ii)
                for (jj = j; jj < je; ++jj)
                    m[ii][jj] += sdot_sse(n_a_cols, a[ii], bT[jj]);
        }
    }
    free_matrix(bT, n_b_cols);
    return m;
}
