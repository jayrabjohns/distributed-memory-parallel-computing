#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "relaxation.h"

#define ROOT 0
#define DEFAULT_TAG 99

int sum(int start, int stop, int step)
{
    int result = 0;
    for (int i = start; i <= stop; i += step)
    {
        printf("[%d] adding %d \n", start, i);
        result += i;
    }

    return result;
}

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

    // Calculate number of rows each procedure will operate on
    int rows_per_proc = p_size / n_procs;
    int rows_left = p_size % n_procs;
    int send_counts[n_procs], displacements[n_procs], row_counts[n_procs];
    int total_displacement = 0;
    for (int i = 0; i < n_procs; i++)
    {
        row_counts[i] = (i < rows_left) ? (rows_per_proc + 1) : rows_per_proc;
        send_counts[i] = row_counts[i] * p_size;
        displacements[i] = total_displacement;
        total_displacement += send_counts[i];
    }

    // Debug logs
    if (rank == ROOT)
    {
        printf("rows per proc %d \n", rows_per_proc);
        printf("remainder rows %d \n", rows_left);
        printf("row counts: ");
        for (size_t i = 0; i < (size_t)n_procs; i++)
        {
            printf("%d, ", send_counts[i]);
        }
        printf("\n");
        printf("row starts: ");
        for (size_t i = 0; i < (size_t)n_procs; i++)
        {
            printf("%d, ", displacements[i]);
        }
        printf("\n");
    }

    // Define local problem to work on
    int rows_count = row_counts[rank];
    double(*problem_local)[rows_count][p_size];
    rc = array_2d_try_alloc((size_t)rows_count, (size_t)p_size, &problem_local);
    if (rc != 0)
        return rc;

    // Distribute work among processes
    MPI_Scatterv((double *)problem_global, send_counts, displacements,
                 MPI_DOUBLE, (double *)problem_local, send_counts[rank],
                 MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    printf("[%d] problem recieved:\n", rank);
    array_2d_print(rows_count, p_size, problem_local, rank);

    // int num_workers = num_procs - 1;
    // if (rank == ROOT)
    // {
    //     // Block until all results are in
    //     printf("number of processes (including root): %d \n", num_procs);
    //     printf("summing 1 -> %d \n", sum_until);

    //     int total = 0;
    //     for (int process = 1; process <= num_workers; process++)
    //     {
    //         int res;
    //         MPI_Recv(&res, 1, MPI_INT, process, DEFAULT_TAG,
    //                  MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //         total = total + res;
    //     }

    //     printf("Result: %d \n", total);
    //     fflush(stdout);
    // }
    // else
    // {
    //     // Calculate and send result to root
    //     int result = sum(rank, sum_until, num_workers);
    //     MPI_Send(&result, 1, MPI_INT, ROOT, DEFAULT_TAG, MPI_COMM_WORLD);
    // }

    MPI_Finalize();
    return 0;
}

// void distribute_work(size_t size, double (*matrix)[size][size], int n_procs)
// {
//     int rows_per_proc = size / n_procs;
//     int remainder_rows = size % n_procs;
//     int proc_rows[n_procs];
//     for (size_t i = 0; i < n_procs; i++)
//     {
//         proc_rows[i] = rows_per_proc;
//         if (i < remainder_rows)
//             proc_rows[i] += 1;
//     }
// }

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
            (*matrix)[i][j] = 0.0;
        }
    }
}
