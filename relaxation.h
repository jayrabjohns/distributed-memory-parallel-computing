#ifndef RELAXATION_H
#define RELAXATION_H

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
A helper function to allocate memory for a 2D square matrix.
It guarantees that memory is allocated as one contiguous block, allowing
stdlib functions such as memcmp & memcpy can be used with the resulting array.

Returns 0 if the allocation is successful.

`size` is the length of one side of `matrix`.
`matrix` will be allocated a pointer for a square 2D array of length `size`.
*/
int array_2d_try_alloc(size_t rows, size_t cols, double (**matrix)[rows][cols]);

/*
A helper function to print a square 2D array to a file or output stream.

`size` is the length of one side of `matrix`.
`matrix` points to a square 2D array of length `size`
`rank` is used to determine from which process this log originates
*/
void array_2d_print(int rows, int cols, double (*matrix)[rows][cols], int rank);

/*
Creates a nxn matrix of the form
[1 1 ... 1]
[1 0 ... 0]
[.........]
[1 0 ... 0]

`size` is the length of one side of `matrix`
`matrix` points to the 2D square array to store this matrix in.
 */
void load_testcase_1(int size, double (*matrix)[size][size]);

#endif