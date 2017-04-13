#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

int main()
{
  size_t bag_rows = 655468, bag_cols = 3, bag_offset = 3;
  int bag[bag_rows][bag_cols];

  printf("Create bag of words\n");
  read_file("../docword.nytimes.txt", bag_offset, bag_rows, bag_cols, bag);

  return 0;
}
