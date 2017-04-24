#include <gsl/gsl_linalg.h>
#include "trabalho_lib.h"

int main()
{
    float** bag = create_bag();
    printf("Bag created...\n");

    float** distances = load_distance();

    float **svd_distances;

    gsl_matrix* A = gsl_matrix_calloc(N_VOCAB, N_DOCS);
    gsl_matrix* V = gsl_matrix_calloc(N_DOCS, N_DOCS);
    gsl_matrix* X = gsl_matrix_calloc(N_DOCS, N_DOCS);
    gsl_vector* S = gsl_vector_calloc(N_DOCS);
    gsl_vector* work = gsl_vector_calloc(N_DOCS);

    // Initialize gsl values
    for (int i = 0; i < N_DOCS; i++)
        for (int j = 0; j < N_VOCAB; j++)
            gsl_matrix_set(A, j, i, (double) bag[i][j]);

    free_matrix(bag, N_DOCS);

    printf("Running SVD...\n");

    clock_t begin = clock();
    gsl_linalg_SV_decomp_mod(A, X, V, S, work);

    printf("SVD time: %f\n", (double) (clock() - begin) / CLOCKS_PER_SEC);

    float** matrix = mat_alloc(N_DOCS, N_DOCS);
    for (int i = 0; i < N_DOCS; i++)
        for (int j = 0; j < N_DOCS; j++)
            matrix[i][j] = (float) gsl_matrix_get(A, i, j);

    gsl_matrix_free(A);
    gsl_matrix_free(V);
    gsl_matrix_free(X);
    gsl_vector_free(work);

    printf("S: ");
    for(int i = 0; i < 70; i++)
        printf("%f ", gsl_vector_get(S, i));
    printf("\n");

    gsl_vector_free(S);

    int dimensions[] = {4, 16, 64, 256};
    for (int idim = 0; idim < 4; idim++)
    {
        float distortion;

        int dim = dimensions[idim];
        printf("Number of dimensions: %d\n\n", dim);

        begin = clock();
        svd_distances = create_distance(matrix, dim);
        printf("Calculate distances (seconds): %f\n", (double) (clock() - begin) / CLOCKS_PER_SEC);

        distortion = max_distortion(distances, svd_distances);
        printf("Max distortion: %f\n", distortion);

        distortion = avg_distortion(distances, svd_distances);
        printf("Avg distortion: %f\n", distortion);

        fflush(stdout);
        free_matrix(svd_distances, N_DOCS);
    }


    free_matrix(distances, N_DOCS);
    free_matrix(matrix, N_DOCS);
    return 1;
}
