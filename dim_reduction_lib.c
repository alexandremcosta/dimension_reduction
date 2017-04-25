#include "dim_reduction_lib.h"

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

    if (bag[0][17653] != 4)
        printf("Bag error\n");

    return bag;
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

float max_distortion(float** x, float** wx)
{
    float distortion, result = 0;

    for (int i = 0; i < N_DOCS; i++)
        for (int j = i+1; j < N_DOCS; j++)
        {
            distortion = fabs((wx[i][j] / x[i][j]) - 1);
            if (distortion > result && fabs(wx[i][j] - 0) > 0.00001 && fabs(x[i][j] - 0) > 0.00001 ) // Ignore distortion on duplicated documents
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
            if (distortion > result && fabs(wx[i][j] - 0) > 0.00001 && fabs(x[i][j] - 0) > 0.00001 ) // Ignore distortion on duplicated documents
            {
                result += distortion;
                count++;
            }
        }

    return result / count;
}

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

void print_time(clock_t begin, clock_t end)
{
    float elapsed = (float)(end - begin) / CLOCKS_PER_SEC;
    printf("%f\t", elapsed);
}

float** load_distance()
{
    float** table;

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

    return table;
}

float** mat_alloc(int rows, int cols)
{
    float** space = malloc(rows * sizeof(float*));
    check_memory(space);
    for (int i = 0; i < rows; i++)
    {
        space[i] = calloc(cols, sizeof(float));
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

// Intel header for faster float operations
#include <xmmintrin.h>

// Adapted from: https://github.com/attractivechaos/matmul
// method: SSE+tiling sdot
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

