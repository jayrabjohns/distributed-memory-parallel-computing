#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "relaxation.h"

#define ROOT 0
#define DEFAULT_TAG 99

// int sum(int start, int stop, int step)
// {
//     int result = 0;
//     for (int i = start; i <= stop; i += step)
//     {
//         printf("[%d] adding %d \n", start, i);
//         result += i;
//     }

//     return result;
// }

void distribute_workload(int p_size, int n_procs, int send_counts[n_procs],
                         int displacements[n_procs], int row_counts[n_procs]);

int main(int argc, char *argv[])
{
    // Init MPI
    int rank, n_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Parse arguments
    if (argc < 3)
    {
        fprintf(stderr, "Usage: relaxation [int problem_size] [float precision] \n");
        return 1;
    }
    int p_size = atoi(argv[1]);
    double precision = atof(argv[2]);

    if (p_size < n_procs + 2)
    {
        // Early fail because this was most almost certainly a mistake
        fprintf(stderr, "The problem size cannot be less than the number of processors + 2");
        return 1;
    }

    // Set up problem
    int rc;
    double(*problem_global)[p_size][p_size];
    rc = array_2d_try_alloc((size_t)p_size, (size_t)p_size, &problem_global);
    if (rc != 0)
        return rc;

    if (rank == ROOT)
    {
        load_testcase_1(p_size, problem_global);
        printf("\n");
        array_2d_print(p_size, p_size, problem_global, rank);
    }

    int send_counts[n_procs], displs[n_procs], row_counts[n_procs];
    distribute_workload(p_size, n_procs, send_counts, displs, row_counts);

    // Debug log rows each process will operate on
    if (rank == ROOT)
    {
        printf("row counts: ");
        for (size_t i = 0; i < (size_t)n_procs; i++)
        {
            printf("%d, ", row_counts[i]);
        }
        printf("\n");
        printf("row starts: ");
        for (size_t i = 0; i < (size_t)n_procs; i++)
        {
            printf("%d, ", displs[i]);
        }
        printf("\n");
        printf("send counts: ");
        for (size_t i = 0; i < (size_t)n_procs; i++)
        {
            printf("%d, ", send_counts[i]);
        }
        printf("\n");
    }

    // Allocate local chunk of the problem to work on
    int n_rows = row_counts[rank];
    double(*local_problem)[n_rows][p_size];
    rc = array_2d_try_alloc((size_t)n_rows, (size_t)p_size, &local_problem);
    if (rc != 0)
        return rc;

    // Distribute work among processes
    MPI_Scatterv((double *)problem_global, send_counts, displs,
                 MPI_DOUBLE, (double *)local_problem, send_counts[rank],
                 MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    printf("[%d] problem recieved:\n", rank);
    array_2d_print(n_rows, p_size, local_problem, rank);

    // Local operations
    for (int i = 1; i < n_rows - 1; i++)
    {
        for (int j = 1; j < p_size - 1; j++)
        {
            (*local_problem)[i][j] += 1;
        }
    }

    // MPI_Gatherv((double *)local_problem, send_counts[rank], MPI_DOUBLE,
    //             (double *)problem_global, send_counts, displs, MPI_DOUBLE,
    //             ROOT, MPI_COMM_WORLD);

    if (rank == ROOT)
    {
        printf("Final array: \n");
        array_2d_print(p_size, p_size, problem_global, rank);
    }

    free(problem_global);
    free(local_problem);

    MPI_Finalize();
    return 0;
}

void distribute_workload(int p_size, int n_procs, int send_counts[n_procs],
                         int displacements[n_procs], int row_counts[n_procs])
{
    int rows_per_proc = (p_size - 2) / n_procs;
    int extra_rows = (p_size - 2) % n_procs;
    int rows, total_displacement = 0;
    for (int i = 0; i < n_procs; i++)
    {
        rows = rows_per_proc + (i < extra_rows ? 1 : 0);
        row_counts[i] = rows + 2;
        send_counts[i] = row_counts[i] * p_size;
        displacements[i] = total_displacement;
        total_displacement += rows * p_size;
    }

    // int rows_per_proc = (p_size - 2) / n_procs;
    // int rows_left = (p_size - 2) % n_procs;
    // int send_counts[n_procs], displs[n_procs], row_counts[n_procs];
    // int extra_rows, total_displacement = 0;
    // for (int i = 0; i < n_procs; i++)
    // {
    //     extra_rows = (i < rows_left) ? 1 : 0;
    //     row_counts[i] = rows_per_proc + extra_rows + 2;
    //     send_counts[i] = row_counts[i] * p_size;
    //     displs[i] = total_displacement;
    //     total_displacement += (row_counts[i] - 2) * p_size;
    // }
}

void local_operations(int rows, int cols, double (*local_matrix)[rows][cols])
{
    // Local operations
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            (*local_matrix)[i][j] += 1;
        }
    }
}

int array_2d_try_alloc(size_t rows, size_t cols, double (**matrix)[rows][cols])
{
    *matrix = malloc(rows * cols * sizeof(double));
    if (*matrix == NULL)
    {
        size_t size_in_gb = rows * cols * sizeof(double) / 1000000000;
        fprintf(stderr, "Cannot allocate memory for %ldx%ld matrix. (%ldGB)\n",
                rows, cols, size_in_gb);
        return 1;
    }

    return 0;
}

void array_2d_print(int rows, int cols, double (*matrix)[rows][cols], int rank)
{
    for (int i = 0; i < rows; i++)
    {
        printf("[%d] ", rank);
        for (int j = 0; j < cols; j++)
        {
            printf("%f ", (*matrix)[i][j]);
        }

        printf("\n");
    }
}

void load_testcase_1(int size, double (*matrix)[size][size])
{
    for (int j = 0; j < size; j++)
    {
        (*matrix)[0][j] = 1.0;
    }

    for (int i = 1; i < size; i++)
    {
        (*matrix)[i][0] = 1.0;
        for (int j = 1; j < size; j++)
        {
            (*matrix)[i][j] = i + 1;
        }
    }
}
