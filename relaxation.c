#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

#include "relaxation.h"

#define ROOT 0
#define DEFAULT_TAG 99

void perform_iteration(int n_rows, int n_cols, double matrix[n_rows][n_cols],
                       double prev_matrix[n_rows][n_cols]);

void sync_with_neighbours(int rank, int n_procs, int n_rows, int n_cols,
                          double local_problem[n_rows][n_cols],
                          MPI_Datatype row_t);

int solve_sync(size_t size, double matrix[size][size], double precision);

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

    // Calculate displacements ignoring the first and last rows
    int recv_displs[n_procs];
    for (int i = 0; i < n_procs; i++)
    {
        recv_displs[i] = send_displs[i] + 1;
    }

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

    // printf("[%d] problem recieved:\n", rank);
    // array_2d_print(n_rows, p_size, local_problem, rank);

    bool is_converged = false;
    int iterations = 0;
    while (iterations < 3)
    {
        // Local operations
        perform_iteration(n_rows, p_size, *local_problem, *local_problem_prev);

        printf("[%d] problem after iteration:\n", rank);
        array_2d_print(n_rows, p_size, local_problem, rank);
        fflush(stdout);

        // is_converged = check_is_converged(n_rows, p_size, *local_problem);
        sync_with_neighbours(rank, n_procs, n_rows,
                             p_size, *local_problem, row_t);

        memcpy(local_problem_prev, local_problem, sizeof(*local_problem_prev));
        iterations++;
    }

    // Ignore first row since otherwise it could overwrite another processe's
    // work during reconstruction.
    const double *sendbuf = &((*local_problem)[1][0]);
    MPI_Gatherv(sendbuf, row_counts[rank], row_t,
                problem_global, row_counts, recv_displs, row_t,
                ROOT, MPI_COMM_WORLD);

    if (rank == ROOT)
    {
        printf("Final array: \n");
        array_2d_print(p_size, p_size, problem_global, rank);
    }

    // free(problem_global);
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

void send_to_neighbours(int rank, int n_procs, const double *send_top, const double *send_bot, MPI_Datatype row_t)
{
    // Send to neighbour above
    if (rank > 0)
        MPI_Send(send_top, 1, row_t, rank - 1, DEFAULT_TAG, MPI_COMM_WORLD);

    // Send to neighbour below
    if (rank < n_procs - 1)
        MPI_Send(send_bot, 1, row_t, rank + 1, DEFAULT_TAG, MPI_COMM_WORLD);
}

void receive_from_neighbours(int rank, int n_procs, const double *recv_top, const double *recv_bot, MPI_Datatype row_t)
{
    // Receive from neighbour above
    if (rank > 0)
    {
        MPI_Status *status = malloc(sizeof(MPI_Status));
        MPI_Recv(recv_top, 1, row_t, rank - 1, DEFAULT_TAG, MPI_COMM_WORLD, status);
    }

    // Receive from neighbour below
    if (rank < n_procs - 1)
    {
        MPI_Status *status = malloc(sizeof(MPI_Status));
        MPI_Recv(recv_bot, 1, row_t, rank + 1, DEFAULT_TAG, MPI_COMM_WORLD, status);
    }
}

/*
barrier
have a group of even rank and group of odd rank.
even rank receives
then odd rank receives

- > very synchronous, would need all processes to be ready to exchange at the same time.
*/
void sync_with_neighbours(int rank, int n_procs, int n_rows, int n_cols,
                          double local_problem[n_rows][n_cols],
                          MPI_Datatype row_t)
{
    // To continue, the neighbours are required to have finished, so this
    // uses a blocking call. We block send as well to simplify implementation.

    // ~~~~ <- recv_top, we receive this from neighbour above
    // #### <- send_top, we send this to neighbour above
    // ####
    // #### <- bot row, we send this to neighbour below
    // @@@@ <- recv_bot, we receive this from neighbour below

    // Even ranks send first and receive second
    // while odd ranks receive first and send second
    const double *send_top = &(local_problem[1][0]);
    const double *send_bot = &(local_problem[n_rows - 2][0]);
    double *recv_top = &(local_problem[0][0]);
    double *recv_bot = &(local_problem[n_rows - 1][0]);

    if (rank % 2 == 0)
    {
        send_to_neighbours(rank, n_procs, send_top, send_bot, row_t);
        receive_from_neighbours(rank, n_procs, recv_top, recv_bot, row_t);
    }
    else
    {
        receive_from_neighbours(rank, n_procs, recv_top, recv_bot, row_t);
        send_to_neighbours(rank, n_procs, send_top, send_bot, row_t);
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

bool matrix_has_converged(double precision, int n_rows, int n_cols,
                          double m1[n_rows][n_cols],
                          double m2[n_rows][n_cols])
{
    bool is_converged = true;
    for (int row = 0; row < n_rows && is_converged; row++)
    {
        for (int col = 0; col < n_cols && is_converged; col++)
        {
            double diff = fabs(m1[row][col] - m2[row][col]);
            if (diff > precision)
                is_converged = false;
        }
    }
    return is_converged;
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

// int test_and_compare(
//     size_t size,
//     double precision,
//     int thread_count,
//     void (*load_test_data)(size_t, double (*)[size][size]))
// {
//     printf("matrix size: %ld\nprecision: %f\nthread count:%ld\n",
//            size, precision, thread_count);

//     // Arrange - allocate memory, initialise variables
//     int rc = 0;
//     double(*result_async)[size][size];
//     double(*result_sync)[size][size];

//     rc = array_2d_try_alloc(size, size, &result_async);
//     if (rc != 0)
//         return rc;

//     rc = array_2d_try_alloc(size, size, &result_sync);
//     if (rc != 0)
//     {
//         free(result_async);
//         return rc;
//     }

//     (*load_test_data)(size, result_async);
//     (*load_test_data)(size, result_sync);

//     // Act - perform the tested action
//     solve(size, result_async, thread_count, precision);
//     solve_sync(size, result_sync, precision);

//     // Assert - assert that the action performed successfully

//     matrix_has_converged(precision, size, size, *result_async, *result_sync);

//     if (rc == 0)
//     {
//         printf("PASS solution matches synchronous implementation\n");
//     }
//     else
//     {
//         rc = 1;
//         printf("FAIL solution doesn't match synchronous implementation\n");
//         printf("\nsync impl result\n");
//         array_2d_print(size, size, result_sync, ROOT);
//         printf("async impl result:\n");
//         array_2d_print(size, size, result_async, ROOT);
//     }

//     free(result_sync);
//     free(result_async);

//     return rc;
// }

int solve_sync(
    size_t size,
    double matrix[size][size],
    double precision)
{
    int rc = 0;
    double(*prev_matrix)[size][size];

    // Allocate memory for copy of previous iteration.
    rc = array_2d_try_alloc(size, size, &prev_matrix);
    if (rc != 0)
        return rc;

    // Copy the previous iteration
    memcpy(prev_matrix, matrix, sizeof(*prev_matrix));
    printf("\n");

    int iterations = 0;
    bool converged = false;
    double elapsed_time;
    time_t start, now;
    time(&start);
    while (!converged)
    {
        perform_iteration(size - 1, size - 1, matrix, *prev_matrix);

        // Check for convergence
        converged = matrix_has_converged(precision, size, size, matrix, prev_matrix);
        memcpy(prev_matrix, matrix, sizeof(*prev_matrix));
        ++iterations;
    }

    time(&now);
    elapsed_time = difftime(now, start);
    printf("[SYNC] solved in %d iterations in %.0lfs\n",
           iterations, elapsed_time);

    // Cleanup
    free(prev_matrix);

    return rc;
}