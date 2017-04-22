#include "svd.h"
#include "trabalho_lib.h"

int main()
{
    float** bag = create_bag();
    printf("Bag created...\n");
    float** transposed = mat_transpose(N_DOCS, N_VOCAB, bag);
    printf("Bag transposed...\n");
    free_matrix(bag, N_DOCS);

    float** distances = load_distance();

    float *d = malloc(N_DOCS * sizeof(float)),
          **v = mat_alloc(N_DOCS, N_DOCS),
          **svd_distances;

    printf("Running SVD...\n");

    clock_t begin = clock();
    dsvd(transposed, N_VOCAB, N_DOCS, d, v);
    printf("SVD time: %f\n", (double) (clock() - begin) / CLOCKS_PER_SEC);

    free_matrix(transposed, N_VOCAB);

    float** matrix = mat_transpose(N_DOCS, N_DOCS, v);
    free_matrix(v, N_DOCS);

    printf("S: ");
    for(int i = 0; i < 70; i++)
        printf("%f ", d[i]);
    printf("\n");
    free(d);

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
