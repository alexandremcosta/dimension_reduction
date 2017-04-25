## Description

Dimension reduction implemented in C.

1. trabalho_rpj.c implements dimension reduction using Random Projections (achlioptas and gaussian matrix)
2. trabalho_svd.c implements dimension reduction using SVD
3. trabalho_lib.c (and .h) is a small library used by 1 and 2 to perform common operations such as matrix multiplication, distance calculation, time measurement, memory allocation and file I/O

The reduced matrix is then compared to the original via euclidian distance.
Maximum and average distortion between documents is also measured.

## Tests

The tests ran on a Linode machine with:
- Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz (2 cores)
- 32 GB
- Arch Linux

Thats because the reduced matrix of a random projection of 15k dimensions consumes more than 8GB of memory, which was the maximum available on local machines.
This infrastructure is capable to run half of the tests in one core, and another half in another core, which generates results twice as fast.

## How to run

1. Download input and extract to this same folder: `docword.nytimes.txt` from https://archive.ics.uci.edu/ml/datasets/Bag+of+Words

2. Compile and run trabalho_rpj.c
```
  $ gcc -lm -lgsl -lgslcblas -Wall trabalho_rpj.c trabalho_lib.c -o trabalho_rpj
  $ ./trabalho_rpj
```
It takes more than 40 minutes to calculate the distances, so be patient.

3. Install SVD dependency: GSL
Ubuntu: `sudo apt-get install libgsl0ldbl`
Arch Linux: `sudo pacman -S gsl`

4. Compile and run trabalho_svd.c
```
  $ gcc -lm -lgsl -lgslcblas -Wall trabalho_svd.c trabalho_lib.c -o trabalho_svd
  $ ./trabalho_svd
```

## Running faster

If you want to run faster, you can reduce the number of documents by commenting this lines from trabalho_lib.h
```
  #define N_DOCS    3000
  #define DOC_LINES 655468
```
And uncommenting the respective lines with the number of dimensions you want. Don't forget to uncomment the DOC_LINES below.
