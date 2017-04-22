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
float** create_bag();
float** create_distance(float**, int);
float   calculate_distance(float*, float*, int);
float** create_achlioptas(int);
float** create_gaussian(int);
float** create_projection(float**, float**, int);
float   max_distortion(float**, float**);
float   avg_distortion(float**, float**);

// Helpers
void    read_file(char[], size_t, size_t, size_t, int[][DOC_COLS]);
void    print_time(clock_t, clock_t);
float** load_distance(float**);
void    save_distance(float** table);
float   rand_normal(float, float);
float** mat_alloc(int, int);
void    free_matrix(float**, int);
void    check_memory(void *);
float** mat_mult(int, int, float *const *, int, float *const *);

int main()
{
    float **rand_matrix, **rand_projection, **rand_distances, distortion;

    float** bag = create_bag();
    float** distances = load_distance(bag);

    clock_t begin;
    srand(time(NULL));

    int dimensions[] = {4, 16, 64, 256, 1024, 4096, 15768};
    for (int idim = 0; idim < 6; idim++)
    {
        int dim = dimensions[idim];
        printf("Number of dimensions: %d\n", dim);
        fflush(stdout);

        printf("Achlioptas\tMatrix\t\tProjection\tDistances\tMax distortion\tAvg distortion\n");
        for (int i = 1; i <= 30; i++)
        {
            printf("%d\t\t", i);

            // Achlioptas
            begin = clock();
            rand_matrix = create_achlioptas(dim);
            print_time(begin, clock());

            begin = clock();
            rand_projection = create_projection(bag, rand_matrix, dim);
            print_time(begin, clock());
            free_matrix(rand_matrix, N_VOCAB);

            begin = clock();
            rand_distances = create_distance(rand_projection, dim);
            print_time(begin, clock());
            free_matrix(rand_projection, N_DOCS);

            distortion = max_distortion(distances, rand_distances);
            printf("%f\t", distortion);
            distortion = avg_distortion(distances, rand_distances);
            printf("%f\n", distortion);
            free_matrix(rand_distances, N_DOCS);

            fflush(stdout);
        }

        printf("\nNormal\t\tMatrix\t\tProjection\tDistances\tMax distortion\tAvg distortion\n");
        for (int i = 1; i <= 30; i++)
        {
            printf("%d\t\t", i);

            // Normal
            begin = clock();
            rand_matrix = create_gaussian(dim);
            print_time(begin, clock());

            begin = clock();
            rand_projection = create_projection(bag, rand_matrix, dim);
            print_time(begin, clock());
            free_matrix(rand_matrix, N_VOCAB);

            begin = clock();
            rand_distances = create_distance(rand_projection, dim);
            print_time(begin, clock());
            free_matrix(rand_projection, N_DOCS);

            distortion = max_distortion(distances, rand_distances);
            printf("%f\t", distortion);
            distortion = avg_distortion(distances, rand_distances);
            printf("%f\n", distortion);
            free_matrix(rand_distances, N_DOCS);

            fflush(stdout);
        }

        printf("---\n\n");
    }

    free_matrix(distances, N_DOCS);
    free_matrix(bag, N_DOCS);
    return 0;
}

float** create_bag()
{
    size_t docword_rows = DOC_LINES, docword_cols = DOC_COLS, docword_offset = 3; // small
    int docword[docword_rows][docword_cols];
    read_file(FILENAME, docword_offset, docword_rows, docword_cols, docword);

    float** bag = mat_alloc(N_DOCS, N_VOCAB);
    for (int i = 0; i < N_DOCS; i++)
      for (int j = 0; j < N_VOCAB; j++)
        bag[i][j] = 0;

    for ( int i = 0; i < docword_rows; i++ )
        bag[docword[i][0]-1][docword[i][1]-1] = (int) docword[i][2];

    (bag[0][17653] == 4) ? printf("Bag created\n") : printf("Bag error\n");

    return bag;
}

float** create_achlioptas(int size)
{
    float** matrix = mat_alloc(N_VOCAB, size);
    for (int i = 0; i < N_VOCAB; i++)
      for (int j = 0; j < size; j++)
      {
          float random = (float)rand() / (float)RAND_MAX;
          random = (random < (1./6) ? -1 : (random < (5./6) ? 0 : 1));
          random = random * sqrt(3./N_VOCAB);
          matrix[i][j] = sqrt(N_VOCAB*1.0/size) * random;
      }

    return matrix;
}

float** create_gaussian(int size)
{
    float** matrix = mat_alloc(N_VOCAB, size);
    for (int i = 0; i < N_VOCAB; i++)
      for (int j = 0; j < size; j++)
      {
          matrix[i][j] = sqrt(N_VOCAB*1.0/size) * rand_normal(0, 1/sqrt(N_VOCAB));
      }

    return matrix;
}

float** create_distance(float** table, int n_cols)
{
    float** distances = mat_alloc(N_DOCS, N_DOCS);
    float distance;

    for (int i = 0; i < N_DOCS; i++)
        for (int j = i+1; j < N_DOCS; j++)
        {
            distance = calculate_distance(table[i], table[j], n_cols);
            distances[i][j] = distance;
        }

    return distances;
}

float calculate_distance(float* a, float* b, int n_cols)
{
    float sum = 0;
    for ( int i = 0; i < n_cols; i++ )
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    return sqrt(sum);
}

float** create_projection(float** bag, float** matrix, int dim)
{
    return mat_mult(N_DOCS, N_VOCAB, bag, dim, matrix);
}

float max_distortion(float** x, float** wx)
{
    float distortion, result = 0;

    for (int i = 0; i < N_DOCS; i++)
        for (int j = i+1; j < N_DOCS; j++)
        {
            distortion = fabs((wx[i][j] / x[i][j]) - 1);
            if (distortion > result && fabs(wx[i][j] - 0) > 0.00001) // Ignore distortion os duplicated documents
            {
                result = distortion;
            }
        }

    return result;
}

float avg_distortion(float** x, float** wx)
{
    float distortion, result = 0;
    int count = 0;

    for (int i = 0; i < N_DOCS; i++)
        for (int j = i+1; j < N_DOCS; j++)
        {
            distortion = fabs((wx[i][j] / x[i][j]) - 1);
            if (distortion > result && fabs(wx[i][j] - 0) > 0.00001) // Ignore distortion os duplicated documents
            {
                result += distortion;
                count++;
            }
        }

    return result / count;
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

float** load_distance(float** bag)
{
    float** table;

    if (0)
    {
        clock_t begin = clock();
        table = create_distance(bag, N_VOCAB);
        printf("Distance calculation: %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
        save_distance(table);
    }
    else {
        FILE *file = fopen("distances.txt", "r");
        table = mat_alloc(N_DOCS, N_DOCS);
        if ( file )
        {
            size_t i, j;
            for ( i = 0; i < N_DOCS; ++i )
                for ( j = 0; j < N_DOCS; ++j )
                    fscanf(file, "%f ", &table[i][j]);
            fclose(file);
        }
    }
    return table;
}

void save_distance(float** table)
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

float** mat_alloc(int rows, int cols)
{
  float** space = malloc(rows * sizeof(float*));
  check_memory(space);
  for (int i = 0; i < rows; i++)
  {
    space[i] = malloc(cols * sizeof(float));
    check_memory(space[i]);
  }
  return space;
}

void free_matrix(float** matrix, int rows)
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

void print_time(clock_t begin, clock_t end)
{
    float elapsed = (float)(end - begin) / CLOCKS_PER_SEC;
    printf("%f\t", elapsed);
}

// https://phoxis.org/2013/05/04/generating-random-numbers-from-normal-distribution-in-c/
float rand_normal(float mu, float sigma)
{
    float U1, U2, W, mult;
    static float X1, X2;
    static int call = 0;

    if (call == 1)
    {
        call = !call;
        return (mu + sigma * (float) X2);
    }

    do
    {
        U1 = -1 + ((float) rand () / RAND_MAX) * 2;
        U2 = -1 + ((float) rand () / RAND_MAX) * 2;
        W = pow (U1, 2) + pow (U2, 2);
    }
    while (W >= 1 || W == 0);

    mult = sqrt ((-2 * log (W)) / W);
    X1 = U1 * mult;
    X2 = U2 * mult;

    call = !call;

    return (mu + sigma * (float) X1);
}

// Intel header for faster float operations
#include <xmmintrin.h>

float sdot_sse(int n, const float *x, const float *y)
{
	int i, n8 = n>>3<<3;
	__m128 vs1, vs2;
	float s, t[4];
	vs1 = _mm_setzero_ps();
	vs2 = _mm_setzero_ps();
	for (i = 0; i < n8; i += 8) {
		__m128 vx1, vx2, vy1, vy2;
		vx1 = _mm_loadu_ps(&x[i]);
		vx2 = _mm_loadu_ps(&x[i+4]);
		vy1 = _mm_loadu_ps(&y[i]);
		vy2 = _mm_loadu_ps(&y[i+4]);
		vs1 = _mm_add_ps(vs1, _mm_mul_ps(vx1, vy1));
		vs2 = _mm_add_ps(vs2, _mm_mul_ps(vx2, vy2));
	}
	for (s = 0.0f; i < n; ++i) s += x[i] * y[i];
	_mm_storeu_ps(t, vs1);
	s += t[0] + t[1] + t[2] + t[3];
	_mm_storeu_ps(t, vs2);
	s += t[0] + t[1] + t[2] + t[3];
	return s;
}

// Matrix multiplication
float** mat_transpose(int rows, int cols, float *const* a)
{
	float** m;
	m = mat_alloc(cols, rows);

	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			m[j][i] = a[i][j];

	return m;
}

float **mat_mult(int n_a_rows, int n_a_cols, float *const *a, int n_b_cols, float *const *b)
{
	int i, j, ii, jj, x = 16, n_b_rows = n_a_cols;
	float **m, **bT;
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
