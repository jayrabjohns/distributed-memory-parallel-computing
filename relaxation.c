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

int main(int argc, char *argv[])
{
    // Init MPI
    int rank, n_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Barrier to synchronise start timer
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Check there are enough arguments
    if (argc < 3)
    {
        fprintf(stderr, "Not enough arguments. \n");
        fprintf(stderr,
                "Usage: relaxation [int problem_size] [float precision] \n");
        return 1;
    }

    // Parse arguments
    int p_size = atoi(argv[1]);
    double precision = atof(argv[2]);

    // Because of assumptions made later, bordering rows are treated as unique,
    //   so the program asserts that there are at least 5 rows per process
    if (p_size < n_procs * 5)
    {
        // Early fail because this was most almost certainly a mistake
        fprintf(stderr,
                "Usage: relaxation [int problem_size] [float precision] \n");
        fprintf(stderr,
                "The problem size cannot be less than the num of nodes * 5 \n");
        return 1;
    }

    // Allocate the global problem. This creates a 2D array represented by
    //   a single contiguous block, which allows casting to a 1D array later on.
    int rc;
    double(*problem_global)[p_size][p_size];
    rc = array_2d_try_alloc((size_t)p_size, (size_t)p_size, &problem_global);
    if (rc != 0)
        return rc;

    // Initialise problem. This could be changed to read in a file or something.
    if (rank == ROOT)
    {
        load_testcase_1(p_size, problem_global);
    }

    // Fairly allocate each process a nearly eaqual chunk of the problem
    int send_counts[n_procs], send_displs[n_procs], row_counts[n_procs];
    distribute_workload(p_size, n_procs, send_counts, send_displs, row_counts);

    // The number of rows this process will locally work on
    int n_rows = send_counts[rank];

    // Calculate displacements ignoring the first and last rows
    //   This is needed for when nodes communicate their finished workloads.
    //   Skipping this would potentially allow nodes to ovewrite others' work
    //   at the end.
    int recv_displs[n_procs];
    for (int i = 0; i < n_procs; i++)
    {
        recv_displs[i] = send_displs[i] + 1;
    }

    // Debug log shows what each process will operate on
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

        printf("size of row: %ldb \n", (size_t)p_size * sizeof(double));
    }

    // Allocate memory for local problem
    double(*local_problem)[n_rows][p_size];
    rc = array_2d_try_alloc((size_t)n_rows, (size_t)p_size, &local_problem);
    if (rc != 0)
        return rc;

    // Allocate memory for the previous iteeration of the local problem
    double(*local_problem_prev)[n_rows][p_size];
    rc = array_2d_try_alloc((size_t)n_rows, (size_t)p_size, &local_problem_prev);
    if (rc != 0)
        return rc;

    // A datatype to represent rows of contiguous values.
    //   This can help conceptually when sending differing amounts of data to
    //   processes, which happens when (p_size - 2) isn't a multiple of n_procs.
    MPI_Datatype row_t;
    MPI_Type_contiguous(p_size, MPI_DOUBLE, &row_t);
    MPI_Type_commit(&row_t);

    // Distribute work among processes
    MPI_Scatterv(problem_global, send_counts, send_displs,
                 row_t, local_problem, send_counts[rank],
                 row_t, ROOT, MPI_COMM_WORLD);

    // Create a copy of the local problem just received from the scatter.
    memcpy(local_problem_prev, local_problem, sizeof(*local_problem_prev));

    // Defining this process' neighbours. Using MPI_PROCC_NULL helps simplify
    //   the handling of edge nodes with only 1 neighbour. When used as a source
    //   or destination, the communication becomes a no-op.
    int proc_above = (rank >= 1) ? rank - 1 : MPI_PROC_NULL;
    int proc_below = (rank < n_procs - 1) ? rank + 1 : MPI_PROC_NULL;

    // Get pointers to rows which overlap between processes.
    double(*first_row)[1][p_size];
    double(*last_row)[1][p_size];
    double(*top_row)[1][p_size];
    double(*bot_row)[1][p_size];
    double(*last_three_rows)[3][p_size];
    get_rows(p_size, n_rows, *local_problem, &first_row, &last_row, &top_row,
             &bot_row, &last_three_rows);

    // Get pointers to rows in the prev copy
    double(*first_row_prev)[1][p_size];
    double(*last_row_prev)[1][p_size];
    double(*top_row_prev)[1][p_size];
    double(*bot_row_prev)[1][p_size];
    double(*last_three_rows_prev)[3][p_size];
    get_rows(p_size, n_rows, *local_problem_prev, &first_row_prev,
             &last_row_prev, &top_row_prev, &bot_row_prev,
             &last_three_rows_prev);

    // Perform local operations until converged
    bool has_converged = false;
    int iterations = 0;
    while (!has_converged)
    {
        // The matrix is composed like this:
        // fffffff <- first_row, we receive this from neighbour above
        // ttttttt <- top_row, we send this to neighbour above
        // #######
        // ####### <- rest of the matrix
        // #######
        // bbbbbbb <- bot_row, we send this to neighbour below
        // lllllll <- last_row, we receive this from neighbour below

        // Compute the top and bottom row first, so we can start
        //   non-blocking communications earlier
        for (int i = 0; i < p_size; i++)
        {
            perform_iteration(3, p_size, *first_row, *first_row_prev);
            perform_iteration(3, p_size, *last_three_rows,
                              *last_three_rows_prev);
        }

        // Send first and last rows
        int request_count = 4;
        MPI_Request reqs[request_count];
        // We are careful to not read or moidfy the buffers used by the send or
        // receive operations before they finish

        // Start receive for neighbouring rows. Copy this into prev_buffer
        //   to avoid conflict while computing the rest of the matrix
        MPI_Irecv(*first_row_prev, 1, row_t, proc_above, DEFAULT_TAG,
                  MPI_COMM_WORLD, &(reqs[0]));
        MPI_Irecv(*last_row_prev, 1, row_t, proc_below, DEFAULT_TAG,
                  MPI_COMM_WORLD, &(reqs[1]));

        // Start send for neighbouring rows. We are careful to not
        //   modify top_row or bot_row until the send has completed.
        MPI_Isend(*top_row, 1, row_t, proc_above, DEFAULT_TAG,
                  MPI_COMM_WORLD, &(reqs[2]));
        MPI_Isend(*bot_row, 1, row_t, proc_below, DEFAULT_TAG,
                  MPI_COMM_WORLD, &(reqs[3]));

        // Compute the rest of the rows.
        perform_iteration(n_rows - 2, p_size, *top_row, *top_row_prev);

        // Check if converged on local problem
        // Ignore first and last rows as they belong to other processes
        has_converged = matrix_has_converged(precision, n_rows - 2, p_size,
                                             *top_row, *top_row_prev);

        // Copy problem to the prev buffer for next iteration. ignoring
        //   first_row and last_row as they will be copied in
        //   by the receive message
        size_t size = sizeof(double) * (size_t)p_size * (size_t)(n_rows - 2);
        memcpy((double *)top_row_prev, (double *)top_row, size);

        // Block to finish communicaiton with neighbours
        MPI_Status statuses[request_count];
        MPI_Waitall(request_count, reqs, statuses);

        // Abort if any messages fail. These will not be caught by
        //   MPI_ERRORS_ARE_FATAL and so must be checked manually
        for (int i = 0; i < request_count; i++)
        {
            int error_code = statuses[i].MPI_ERROR;
            if (error_code != MPI_SUCCESS)
            {
                char err_str[MPI_MAX_ERROR_STRING] = "";
                int result_len;
                MPI_Error_string(error_code, err_str, &result_len);
                fprintf(stderr,
                        "[%d] Aborting. Error sending message to neighbour: %s \n",
                        rank, err_str);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }

        // Check if all processes have converged by reducing convergence flags
        //   with logical-and operation. Send type is an MPI_BYTE because
        //   MPI_LAND doesn't support C booleans
        bool all_have_converged = false;
        MPI_Allreduce(&has_converged, &all_have_converged, 1, MPI_BYTE,
                      MPI_LAND, MPI_COMM_WORLD);

        // Force chunk to continue iterating if flag is false somewhere else
        has_converged = all_have_converged;
        iterations++;
    }

    printf("[%d] total iterations: %d \n", rank, iterations);

    // Ignore first row since otherwise it could overwrite another process'
    //   work during reconstruction.
    const double *sendbuf = &((*local_problem)[1][0]);
    MPI_Gatherv(sendbuf, row_counts[rank], row_t, problem_global, row_counts,
                recv_displs, row_t, ROOT, MPI_COMM_WORLD);

    // Free local resources. The root might still require problem_global for
    //   testing purposes
    free(local_problem);
    MPI_Type_free(&row_t);

    // Synchronise for the final timing calculation
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    MPI_Finalize();

    // Perform unit tests if needed
    if (rank == ROOT)
    {
        printf("Solved in %fs \n", (end_time - start_time));
        test_result_matches_sync_impl(p_size, precision, *problem_global,
                                      &load_testcase_1);

        test_result_has_converged(p_size, problem_global, precision);
    }

    // Free remaining resources
    free(problem_global);

    return 0;
}

/*
Given an m x n matrix this will average each value a_ij with its
4 directly touching neighbours.

The first and last rows of each matrix will be ignored.
E.g. if two 3xn matrices are passed in, only the centre row will have
work done to it.
*/
void perform_iteration(int n_rows, int n_cols, double matrix[n_rows][n_cols],
                       double prev_matrix[n_rows][n_cols])
{
    for (int row = 1; row < n_rows - 1; row++)
    {
        for (int col = 1; col < n_cols - 1; col++)
        {
            double neighbours_sum =
                prev_matrix[row - 1][col] +
                prev_matrix[row + 1][col] +
                prev_matrix[row][col - 1] +
                prev_matrix[row][col + 1];
            matrix[row][col] = 0.25 * neighbours_sum;
        }
    }
}

void distribute_workload(int p_size, int n_procs, int send_counts[n_procs],
                         int displacements[n_procs], int row_counts[n_procs])
{
    int total_rows = (p_size - 2);
    int rows_per_proc = total_rows / n_procs;
    int remainder_rows = total_rows % n_procs;
    int n_rows, total_displacement = 0;
    for (int i = 0; i < n_procs; i++)
    {
        n_rows = rows_per_proc + (i < remainder_rows ? 1 : 0);
        row_counts[i] = n_rows;
        send_counts[i] = n_rows + 2;
        displacements[i] = total_displacement;
        total_displacement += n_rows;
    }
}

void get_rows(int p_size, int n_rows,
              double matrix[p_size][p_size],
              double (**first_row)[1][p_size],
              double (**last_row)[1][p_size],
              double (**top_row)[1][p_size],
              double (**bot_row)[1][p_size],
              double (**last_three_rows)[3][p_size])
{
    // The matrix is composed like this:
    // fffffff <- first_row, we receive this from neighbour above
    // ttttttt <- top_row, we send this to neighbour above
    // #######
    // ####### <- rest of the matrix
    // #######
    // bbbbbbb <- bot_row, we send this to neighbour below
    // lllllll <- last_row, we receive this from neighbour below

    // Because of how matrices are represented in this program,
    //   it's a little tricky to take pointer of it.
    //   Each step has been broken down into taking the pointer of the row,
    //   then casting it to the correct type. The cast step isn't strictly
    //   necessary, but it prevents some warnings when compiled with -Wall

    double *first_row_double = &matrix[0][0];
    *first_row = (double(*)[][p_size])first_row_double;

    double *last_row_double = &matrix[n_rows - 1][0];
    *last_row = (double(*)[][p_size])last_row_double;

    double *top_row_double = &matrix[1][0];
    *top_row = (double(*)[][p_size])(top_row_double);

    double *bot_row_double = &matrix[n_rows - 2][0];
    *bot_row = (double(*)[][p_size])bot_row_double;

    double *last_three_rows_double = &matrix[n_rows - 3][0];
    *last_three_rows = (double(*)[][p_size])last_three_rows_double;
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
            {
                is_converged = false;
            }
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

bool test_result_matches_sync_impl(int p_size, double precision,
                                   double(comparison)[p_size][p_size],
                                   void (*load_testcase)(int, double (*)[p_size][p_size]))
{
    double(*sync_solution)[p_size][p_size];
    array_2d_try_alloc((size_t)p_size, (size_t)p_size, &sync_solution);
    (*load_testcase)(p_size, sync_solution);
    solve_sync(p_size, sync_solution, precision);

    if (p_size <= 20)
    {
        printf("synchronously solved solution: \n");
        array_2d_print(p_size, p_size, sync_solution, ROOT);
    }

    bool sols_match = matrix_has_converged(precision, p_size, p_size,
                                           comparison, *sync_solution);
    if (sols_match)
    {
        printf("PASS Solutions match within a precision of %f \n", precision);
    }
    else
    {
        printf("FAIL Solutions don't match within the given precision (%f) \n",
               precision);
    }

    free(sync_solution);
    return sols_match;
}

bool test_result_has_converged(int p_size, double (*mat)[p_size][p_size],
                               double precision)
{
    double(*mat_copy)[p_size][p_size];
    array_2d_try_alloc((size_t)p_size, (size_t)p_size, &mat_copy);
    memcpy(mat_copy, mat, sizeof(*mat_copy));

    perform_iteration(p_size, p_size, *mat_copy, *mat);
    bool has_converged = matrix_has_converged(precision, p_size, p_size,
                                              *mat_copy, *mat);

    if (has_converged)
    {
        printf(
            "PASS Final matrix has successfully converged with a precision of %f \n",
            precision);
    }
    else
    {
        printf("FAIL Final matrix has not converged with a precision of %f \n",
               precision);
    }

    free(mat_copy);
    return has_converged;
}

int solve_sync(int p_size, double (*matrix)[p_size][p_size], double precision)
{
    // Allocate memory to keep a copy of the previous iteration
    int rc = 0;
    double(*prev_matrix)[p_size][p_size];
    rc = array_2d_try_alloc((size_t)p_size, (size_t)p_size, &prev_matrix);
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
        perform_iteration(p_size, p_size, *matrix, *prev_matrix);
        converged = matrix_has_converged(precision, p_size, p_size, *matrix,
                                         *prev_matrix);
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