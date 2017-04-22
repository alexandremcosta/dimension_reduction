#include "trabalho_lib.h"

// Algorithms
float** create_achlioptas(int);
float** create_gaussian(int);
float** create_projection(float**, float**, int);

// Helpers
void    save_distance(float** table);
float   rand_normal(float, float);

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
    return 1;
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

float** create_projection(float** bag, float** matrix, int dim)
{
    return mat_mult(N_DOCS, N_VOCAB, bag, dim, matrix);
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
