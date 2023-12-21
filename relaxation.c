#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "relaxation.h"

#define ROOT 0
#define DEFAULT_TAG 99

void perform_iteration(int n_rows, int n_cols, double matrix[n_rows][n_cols],
                       double prev_matrix[n_rows][n_cols]);

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

    int send_counts[n_procs], send_displs[n_procs], row_counts[n_procs];
    distribute_workload(p_size, n_procs, send_counts, send_displs, row_counts);

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
            printf("%d, ", send_displs[i]);
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
    int n_rows = send_counts[rank];
    double(*local_problem)[n_rows][p_size];
    rc = array_2d_try_alloc((size_t)n_rows, (size_t)p_size, &local_problem);
    if (rc != 0)
        return rc;

    double(*local_problem_prev)[n_rows][p_size];
    rc = array_2d_try_alloc((size_t)n_rows, (size_t)p_size, &local_problem_prev);
    if (rc != 0)
        return rc;

    // A datatype to represent rows of contiguous values
    MPI_Datatype row_t;
    MPI_Type_contiguous(p_size, MPI_DOUBLE, &row_t);
    MPI_Type_commit(&row_t);

    // Distribute work among processes
    MPI_Scatterv(problem_global, send_counts, send_displs,
                 row_t, local_problem, send_counts[rank],
                 row_t, ROOT, MPI_COMM_WORLD);

    // Create a copy of the matrix
    memcpy(local_problem_prev, local_problem, sizeof(*local_problem_prev));

    printf("[%d] problem recieved:\n", rank);
    array_2d_print(n_rows, p_size, local_problem, rank);

    // Local operations
    perform_iteration(n_rows, p_size, *local_problem, *local_problem_prev);

    // Ignore first row since otherwise it could overwrite another processe's
    // work during reconstruction.
    const double *sendbuf = &((*local_problem)[1][0]);

    // Calculate displacements relative to the global array because of this change
    int recv_displs[n_procs];
    for (int i = 0; i < n_procs; i++)
    {
        recv_displs[i] = send_displs[i] + 1;
    }

    MPI_Gatherv(sendbuf, row_counts[rank], row_t,
                problem_global, row_counts, recv_displs, row_t,
                ROOT, MPI_COMM_WORLD);

    if (rank == ROOT)
    {
        printf("Final array: \n");
        array_2d_print(p_size, p_size, problem_global, rank);
    }

    free(problem_global);
    free(local_problem);

    MPI_Type_free(&row_t);
    MPI_Finalize();
    return 0;
}

void perform_iteration(int n_rows, int n_cols, double matrix[n_rows][n_cols],
                       double prev_matrix[n_rows][n_cols])
{
    // Perform calculations from start row until end row
    for (int row = 1; row < n_rows - 1; row++)
    {
        for (int col = 1; col < n_cols - 1; col++)
        {
            double neighbours_sum =
                prev_matrix[row - 1][col] +
                prev_matrix[row + 1][col] +
                prev_matrix[row][col - 1] +
                prev_matrix[row][col + 1];
            matrix[row][col] = neighbours_sum / 4.0;
        }
    }
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
        row_counts[i] = rows;
        send_counts[i] = rows + 2;
        displacements[i] = total_displacement;
        total_displacement += rows;
    }
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
            (*matrix)[i][j] = 0.0; // i + 1;
        }
    }
}
