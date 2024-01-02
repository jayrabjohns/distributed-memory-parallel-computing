#ifndef RELAXATION_H
#define RELAXATION_H

#include <stdbool.h>

typedef struct
{
    bool has_converged;
    // double *row;
} neighbour_data;

/*
Distributes work evenly among processes. The first and last rows of the matrix
are not worked on, so they are not allocated.

Each process works on at least one row of a matrix to work on.

To do this work it also needs the values from the neighbouring rows above
and below it's chunk.

For this reason when distributing the work, each process receives the number
of rows per process + 2.

When reconstructing the global array, these extra rows
should be omitted as to not overwrite another rows work.
*/
void distribute_workload(int p_size, int n_procs, int send_counts[n_procs],
                         int displacements[n_procs], int row_counts[n_procs]);

/*
Adapted from coursework 1.

Checks if `m1` has converged with resepect to `m2`.
In this case, all elements of `m1` should differ by at most `precision`
from respective elements in `m2`.

Returns true if `m1` has converged, false otherwise.

`precision` is the max difference between iterations to be classed as converged.
`size` is the length of one side of `m1` or `m2`.
`m1` is an `n_rows`x`n_cols` matrix
`m2` is an `n_rows`x`n_cols` matrix
*/
bool matrix_has_converged(double precision, int n_rows, int n_cols,
                          double m1[n_rows][n_cols],
                          double m2[n_rows][n_cols]);

/*
Adapted from coursework 1.

A helper function to allocate memory for a 2D square matrix.
It guarantees that memory is allocated as one contiguous block, allowing
stdlib functions such as memcmp & memcpy can be used with the resulting array.

Returns 0 if the allocation is successful.

`size` is the length of one side of `matrix`.
`matrix` will be allocated a pointer for a square 2D array of length `size`.
*/
int array_2d_try_alloc(size_t rows, size_t cols, double (**matrix)[rows][cols]);

/*
Adapted from coursework 1.

A helper function to print a square 2D array to a file or output stream.

`size` is the length of one side of `matrix`.
`matrix` points to a square 2D array of length `size`
`rank` is used to determine from which process this log originates
*/
void array_2d_print(int rows, int cols, double (*matrix)[rows][cols], int rank);

/*
Adapted from coursework 1.

Creates a nxn matrix of the form
[1 1 ... 1]
[1 0 ... 0]
[.........]
[1 0 ... 0]

`size` is the length of one side of `matrix`
`matrix` points to the 2D square array to store this matrix in.
 */
void load_testcase_1(int size, double (*matrix)[size][size]);

/*
Runs a synchronous implemetation for the given problem and compares
it with a given solution.

Returns true if the solutions match within one unit of `precision`.SUCCESS

`p_size` is the length of one side of `comparison.
`comparison` is a 2D square matrix and is the solution to compare against.
`load_testcase` is a funciton that provides the starting problem configuraion.
*/
bool test_result_matches_sync_impl(int p_size, double precision,
                                   double(comparison)[p_size][p_size],
                                   void (*load_testcase)(int, double (*)[p_size][p_size]));

/*
Adapted from coursework 1.

Performs the relaxation technique on a given matrix synchronously.

Returns 0 if exited normally.

`size` is the length of one side of the matrix.
`matrix` points to a 2D square matrix with length equal to `size`.
`precision` is the max difference between iterations to be classed as converged.
*/
int solve_sync(int p_size, double (*matrix)[p_size][p_size], double precision);

#endif