#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define CALCULATE_DIST 0
#define FILENAME       "docword.nytimes.txt"
#define N_DOCS         3000
#define DOC_LINES      655468
#define DOC_COLS       3
#define N_VOCAB        102660

// Algorithms
double** create_bag();
double** create_distance(double**, int);
double   calculate_distance(double*, double*, int);
double** create_achlioptas(int);
double** create_gaussian(int);
double** create_projection(double**, double**, int);
double   max_distortion(double**, double**);
double   avg_distortion(double**, double**);

// Helpers
void    read_file(char[], size_t, size_t, size_t, int[][DOC_COLS]);
void    print_time(clock_t, clock_t);
double** load_distance(double**);
void    save_distance(double** table);
double   rand_normal(double, double);
double** mat_alloc(int, int);
void    free_matrix(double**, int);
void    check_memory(void *);
double** mat_mult(int, int, double *const *, int, double *const *);

int main()
{
    double **rand_matrix, **rand_projection, **rand_distances, distortion;

    double** bag = create_bag();
    double** distances = load_distance(bag);

    clock_t begin;
    srand(time(NULL));

    int dimensions[] = {4, 16, 64, 256, 1024, 4096, 15768};
    for (int idim = 0; idim < 7; idim++)
    {
        int dim = dimensions[idim];
        printf("Number of dimensions: %d\n\n", dim);
        fflush(stdout);

        printf("Achlioptas\tMatrix (sec)\tProj (sec)\tDistance (sec)\tMax distortion\tAvg distortion\n");
        for (int i = 1; i <= 30; i++)
        {
            printf("%d\t\t", i);

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

        printf("\nNormal\t\tMatrix (sec)\tProj (sec)\tDistance (sec)\tMax distortion\tAvg distortion\n");
        for (int i = 1; i <= 30; i++)
        {
            printf("%d\t\t", i);

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

        printf("\n\n");
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

    if (bag[0][17653] != 4)
        printf("Bag error\n");

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
            matrix[i][j] = sqrt(N_VOCAB*1.0/size) * random;
        }

    return matrix;
}

double** create_gaussian(int size)
{
    double** matrix = mat_alloc(N_VOCAB, size);
    for (int i = 0; i < N_VOCAB; i++)
        for (int j = 0; j < size; j++)
        {
            matrix[i][j] = sqrt(N_VOCAB*1.0/size) * rand_normal(0, 1/sqrt(N_VOCAB));
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
    return mat_mult(N_DOCS, N_VOCAB, bag, dim, matrix);
}

double max_distortion(double** x, double** wx)
{
    double distortion, result = 0;

    for (int i = 0; i < N_DOCS; i++)
        for (int j = i+1; j < N_DOCS; j++)
        {
            distortion = fabs((wx[i][j] / x[i][j]) - 1);
            if (distortion > result && fabs(wx[i][j] - 0) > 0.00001) // Ignore distortion on duplicated documents
            {
                result = distortion;
            }
        }

    return result;
}

double avg_distortion(double** x, double** wx)
{
    double distortion, result = 0;
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

double** load_distance(double** bag)
{
    double** table;

    if (CALCULATE_DIST)
    {
        clock_t begin = clock();
        table = create_distance(bag, N_VOCAB);
        printf("Distance calculation: %f seconds\n", (double)(clock() - begin) / CLOCKS_PER_SEC);
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
                    fscanf(file, "%lf ", &table[i][j]);
            fclose(file);
        }
    }
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

void print_time(clock_t begin, clock_t end)
{
    double elapsed = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("%f\t", elapsed);
}

// https://phoxis.org/2013/05/04/generating-random-numbers-from-normal-distribution-in-c/
double rand_normal(double mu, double sigma)
{
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

// Intel header for faster double operations
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

// Matrix multiplication
double** mat_transpose(int rows, int cols, double *const* a)
{
    double** m;
    m = mat_alloc(cols, rows);

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            m[j][i] = a[i][j];

    return m;
}

double **mat_mult(int n_a_rows, int n_a_cols, double *const *a, int n_b_cols, double *const *b)
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