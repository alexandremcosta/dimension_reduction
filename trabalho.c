#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N_DOCS 3001
#define N_VOCAB 102661

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

int** allocate_bag()
{
    int** bag = malloc(N_DOCS * sizeof(int *));

    for(int i = 0; i < N_DOCS; i++)
    {
        bag[i] = malloc(N_VOCAB * sizeof(int));
        for(int j = 0; j < N_VOCAB; j++)
            bag[i][j] = 0;
    }

    return bag;
}

void free_bag(int** bag)
{
    for(int i = 0; i < N_DOCS; i++)
        free(bag[i]);
    free(bag);
}

int** create_bag()
{
    int** bag = allocate_bag();
    size_t docword_rows = 655468, docword_cols = 3, docword_offset = 3;
    int docword[docword_rows][docword_cols];

    read_file("docword.nytimes.txt", docword_offset, docword_rows, docword_cols, docword);

    for ( int i = 0; i < docword_rows; i++ )
        bag[docword[i][0]][docword[i][1]] = (int) docword[i][2];

    return bag;
}

double dist_euclid(int* a, int* b)
{
    double sum = 0;
    for ( int i = 0; i < N_VOCAB; i++ )
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    return sqrt(sum);
}

void brute_force_search(int** table)
{
    double** distances = malloc(N_DOCS * sizeof(double *));
    for(int i = 0; i < N_DOCS; i++)
        distances[i] = malloc(N_DOCS * sizeof(double));

    clock_t begin = clock();

    for (int i = 1; i < N_DOCS; i++)
    {
        for (int j = i+1; j < N_DOCS; j++)
        {
            distances[i][j] = dist_euclid(table[i], table[j]);
        }
    }

    double time_spent = (double)(clock() - begin) / CLOCKS_PER_SEC;

    printf("Brute force time: %.1f\n minutes", time_spent / 60);

    for(int i = 0; i < N_DOCS; i++)
        free(distances[i]);
    free(distances);
}

int main()
{
    int** bag = create_bag();
    brute_force_search(bag);

    getchar();

    free_bag(bag);
    return 0;
}


      //  Time estimating
      //  double progress = 100.0*i/N_DOCS;
      //  double elapsed = (double)(clock() - begin) / CLOCKS_PER_SEC;
      //  double total = elapsed*N_DOCS/i;
      //  printf("%.5f %% - Elapsed time: %.2f - Remaining time: %.2f minutes \r", progress, elapsed, (total - elapsed)/60); fflush(stdout);

