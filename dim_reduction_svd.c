#include <gsl/gsl_linalg.h>
#include "dim_reduction_lib.h"

int main()
{
    float** bag = create_bag();
    printf("Bag created...\n");

    float ** distances = load_distance();
    float **svd_distances;

    // gsl alloc memory
    gsl_matrix *A, *V, *X;
    gsl_vector *S, *work;
    A    = gsl_matrix_calloc(N_VOCAB, N_DOCS);
    V    = gsl_matrix_calloc(N_DOCS, N_DOCS);
    X    = gsl_matrix_calloc(N_DOCS, N_DOCS);
    S    = gsl_vector_calloc(N_DOCS);
    work = gsl_vector_calloc(N_DOCS);

    // initialize transposed gsl values with bag
    for (int i = 0; i < N_DOCS; i++)
        for (int j = 0; j < N_VOCAB; j++)
            gsl_matrix_set(A, j, i, (double) bag[i][j]);

    printf("Running SVD...\n");
    clock_t begin = clock();
    gsl_linalg_SV_decomp_mod(A, X, V, S, work);
    printf("SVD time: %f\n", (double) (clock() - begin) / CLOCKS_PER_SEC);

    // initialize diagonal matrix with S values, and transpose V
    float **diagonal = mat_alloc(N_DOCS, N_DOCS),
          **v_transp = mat_alloc(N_DOCS, N_DOCS);
    for (int i = 0; i < N_DOCS; i++)
    {
        diagonal[i][i] = (float) gsl_vector_get(S, i);
        for (int j = 0; j < N_DOCS; j++)
            v_transp[j][i] = (float) gsl_matrix_get(V, i, j);
    }

    // print S values
    printf("S: ");
    for(int i = 0; i < N_DOCS/2; i++)
        printf("%f ", diagonal[i][i]);
    printf("\n");

    int dimensions[] = {4, 16, 64, 256};
    for (int idim = 0; idim < 4; idim++)
    {
        float distortion;

        int dim = dimensions[idim];
        printf("Number of dimensions: %d\n", dim);

        // Generate reduced matrix and calculate distances
        begin = clock();
        float **result = mat_mult(dim, dim, diagonal, N_DOCS, v_transp);
        float **matrix = mat_transpose(dim, N_DOCS, result);
        svd_distances = create_distance(matrix, dim);
        printf("Calculate distances (seconds): %f\n", (double) (clock() - begin) / CLOCKS_PER_SEC);

        distortion = max_distortion(distances, svd_distances);
        printf("Max distortion: %f\n", distortion);

        distortion = avg_distortion(distances, svd_distances);
        printf("Avg distortion: %f\n\n", distortion);

        fflush(stdout);

        free_matrix(svd_distances, N_DOCS);
        free_matrix(matrix, N_DOCS);
        free_matrix(result, dim);
    }

    gsl_vector_free(work);
    gsl_vector_free(S);
    gsl_matrix_free(A);
    gsl_matrix_free(V);
    gsl_matrix_free(X);
    free_matrix(bag, N_DOCS);
    free_matrix(distances, N_DOCS);
    free_matrix(v_transp, N_DOCS);
    free_matrix(diagonal, N_DOCS);
    return 1;
}
