#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// #define N_DOCS 3001 // slow
// #define N_LINES 655468 //slow
#define N_DOCS    101 // fast
#define DOC_LINES 21853 // fast
#define DOC_COLS  3
#define N_VOCAB   102661

void   read_file(char[], size_t, size_t, size_t, int[][DOC_COLS]);
void*  create_bag();
double dist_euclid(int[], int[]);
void   brute_force_search(int (*table)[N_VOCAB]);
void*  safe_alloc(int);

int main()
{
    int(*bag)[N_VOCAB] = create_bag();
    brute_force_search(bag); // 42 minutes

    free(bag);
    return 0;
}

void* create_bag()
{
    size_t docword_rows = DOC_LINES, docword_cols = DOC_COLS, docword_offset = 3; // small

    int docword[docword_rows][docword_cols];
    read_file("docword.nytimes.small.txt", docword_offset, docword_rows, docword_cols, docword);


    int (*bag) [N_VOCAB] = safe_alloc(sizeof(int[N_DOCS][N_VOCAB]));
    for ( int i = 0; i < N_DOCS; i++ )
      for ( int j = 0; j < N_VOCAB; j++ )
        bag[i][j] = 0;

    for ( int i = 0; i < docword_rows; i++ )
        bag[docword[i][0]][docword[i][1]] = (int) docword[i][2];

    if ( bag[1][17654] == 4)
      printf("Bag created\n");

    return bag;
}

void brute_force_search(int (*table)[N_VOCAB])
{
    double (*distances) [N_DOCS] = safe_alloc(sizeof(double[N_DOCS][N_DOCS]));

    clock_t begin = clock();

    for (int i = 1; i < N_DOCS; i++)
    {
        // double progress = 100.0*i/N_DOCS;
        // double elapsed = (double)(clock() - begin) / CLOCKS_PER_SEC;
        // double total = elapsed*N_DOCS/i;
        // printf("%.5f%% - Elapsed time: %.4f minutes - Remaining time: %.4f minutes \r", progress, elapsed/60, (total - elapsed)/60); fflush(stdout);

        for (int j = i+1; j < N_DOCS; j++)
        {
            distances[i][j] = dist_euclid(table[i], table[j]);
        }
    }

    double time_spent = (double)(clock() - begin) / CLOCKS_PER_SEC;

    printf("\nBrute force time: %.4f minutes ", time_spent / 60);
    if ( (int) distances[1][2] == 26 )
      printf("OK!\n");
    else
      printf("Error!\n");

    free(distances);
}

double dist_euclid(int a[], int b[])
{
    double sum = 0;
    for ( int i = 0; i < N_VOCAB; i++ )
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    return sqrt(sum);
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

void* safe_alloc(int size)
{
  void* space = malloc(size);
  if (space == NULL)
  {
    printf("Memory allocation error!");
    exit(0);
  }
  return space;
}
