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
#define CONVERGENCE_TAG 98
#define ASYNC_COMMS true

void perform_iteration(int n_rows, int n_cols, double matrix[n_rows][n_cols],
                       double prev_matrix[n_rows][n_cols]);

void sync_with_neighbours(int rank, int n_procs, int n_rows, int n_cols,
                          double local_problem[n_rows][n_cols],
                          MPI_Datatype row_t, bool *has_converged,
                          bool *has_top_converged, bool *has_bot_converged,
                          int iterations);

bool test_result_has_converged(int p_size, double (*mat)[p_size][p_size], double precision);

void check_proc_converged(int rank, int source, bool *has_converged);

int main(int argc, char *argv[])
{
    // Init MPI
    int rank, n_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Parse arguments
    if (argc < 3)
    {
        fprintf(stderr, "Not enough arguments. \n");
        fprintf(stderr, "Usage: relaxation [int problem_size] [float precision] \n");
        return 1;
    }
    int p_size = atoi(argv[1]);
    double precision = atof(argv[2]);

    if (p_size < n_procs * 5)
    {
        // Early fail because this was most almost certainly a mistake
        fprintf(stderr, "Usage: relaxation [int problem_size] [float precision] \n");
        fprintf(stderr, "The problem size cannot be less than the number of processors * 5 \n");
        return 1;
    }

    int rc;
    double(*problem_global)[p_size][p_size];
    rc = array_2d_try_alloc((size_t)p_size, (size_t)p_size, &problem_global);
    if (rc != 0)
        return rc;

    if (rank == ROOT)
    {
        load_testcase_1(p_size, problem_global);
        // printf("\n");
        // array_2d_print(p_size, p_size, problem_global, rank);
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

        printf("size of row: %ldb \n", p_size * sizeof(double));
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

    // MPI_Datatype neighbour_data_t;
    // MPI_Aint startid, startarray;
    // MPI_Get_address(&(buffer.id), &startid);
    // MPI_Get_address(&(buffer.array[0]), &startarray);
    // const int block_legnths[2] = {1, 1};
    // const MPI_Aint displacements[2] = {0, 1};
    // const MPI_Datatype field_types[2] = {MPI_C_BOOL, row_t};

    // const int block_legnths[1] = {1};
    // const MPI_Aint displacements[1] = {0};
    // const MPI_Datatype field_types[1] = {MPI_C_BOOL};

    // MPI_Type_create_struct(1, block_legnths, displacements, field_types,
    //                        &neighbour_data_t);
    // MPI_Type_commit(&neighbour_data_t);

    // Distribute work among processes
    MPI_Scatterv(problem_global, send_counts, send_displs,
                 row_t, local_problem, send_counts[rank],
                 row_t, ROOT, MPI_COMM_WORLD);

    // Create a copy of the matrix
    memcpy(local_problem_prev, local_problem, sizeof(*local_problem_prev));

    // printf("[%d] problem recieved:\n", rank);
    // array_2d_print(n_rows, p_size, local_problem, rank);

    int proc_above = (rank >= 1) ? rank - 1 : MPI_PROC_NULL;
    int proc_below = (rank < n_procs - 1) ? rank + 1 : MPI_PROC_NULL;

    bool has_converged = false, has_top_converged = false, has_bot_converged = false;
    int iterations = 0;
    while (!has_converged)
    {
        // printf("[%d] has_top_converged: %d -- has_bot_converged: %d \n", rank, has_top_converged, has_bot_converged);
        // Local operations

        // Because each of these are assumed to be different,
        // the program asserts that there are at least 5 rows per process
        // on startup

        // ffff <- first_row, we receive this from neighbour above
        // tttt <- top_row, we send this to neighbour above
        // #### <- inner rows
        // ####
        // ####
        // bbbb <- bot_row, we send this to neighbour below
        // llll <- last_row, we receive this from neighbour below
        double *first_row = &((*local_problem)[0][0]);
        double *first_row_prev = &(*local_problem_prev[0][0]);

        double *top_row = &((*local_problem)[1][0]);
        double *top_row_prev = &((*local_problem_prev)[1][0]);

        double *inner_rows = &((*local_problem)[2][0]);
        double *inner_rows_prev = &((*local_problem_prev)[2][0]);

        double *bot_row = &((*local_problem)[n_rows - 2][0]);

        double *last_row = &((*local_problem)[n_rows - 1][0]);
        double *last_row_prev = &((*local_problem_prev)[n_rows - 1][0]);

        double *last_three_rows = &((*local_problem)[n_rows - 3][0]);
        double *last_three_rows_prev = &((*local_problem_prev)[n_rows - 3][0]);

        // Compute first and last row
        // double(*first_row)[1][p_size] = (double(*)[1][p_size])(&((*local_problem)[0][0]));
        // double(*first_row_prev)[1][p_size] = (double(*)[1][p_size])(&((*local_problem_prev)[0][0]));

        // double(*last_row)[1][p_size] = (double(*)[1][p_size])(&((*local_problem)[n_rows - 1][0]));
        // double(*last_row_prev)[1][p_size] = (double(*)[1][p_size])(&((*local_problem_prev)[n_rows - 1][0]));

        int debug_rank = -1;
        // if (rank == debug_rank)
        // {
        //     printf("\n\n");
        //     printf("[%d] [it %d] chunk: \n", rank, iterations);
        //     array_2d_print(n_rows, p_size, local_problem, rank);
        // }

        for (int i = 0; i < p_size; i++)
        {
            perform_iteration(3, p_size, *((double(*)[][p_size])first_row),
                              *((double(*)[][p_size])first_row_prev));
            perform_iteration(3, p_size, *((double(*)[][p_size])last_three_rows),
                              *((double(*)[][p_size])last_three_rows_prev));
        }

        if (rank == debug_rank)
        {
            printf("\n\n");
            printf("[%d] [it %d] chunk after top and bottom rows: \n", rank, iterations);
            array_2d_print(n_rows, p_size, local_problem, rank);
        }

// Send first and last rows
#if ASYNC_COMMS
        int request_count = 2;
        MPI_Request reqs[request_count];
        // Start receive for neighbouring rows
        // Copy this into the prev_buffer to avoid conflict while were computing the rest
        MPI_Irecv(*((double(*)[][p_size])first_row_prev), 1, row_t, proc_above,
                  DEFAULT_TAG, MPI_COMM_WORLD, &(reqs[0]));
        MPI_Irecv(*((double(*)[][p_size])last_row_prev), 1, row_t, proc_below,
                  DEFAULT_TAG, MPI_COMM_WORLD, &(reqs[1]));

        // We are careful to not read or moidfy the buffers used by the send or
        // receive operations before they finish
        // MPI_Isend(*((double(*)[][p_size])top_row), 1, row_t, proc_above,
        //           DEFAULT_TAG, MPI_COMM_WORLD, &(reqs[2]));
        // MPI_Isend(*((double(*)[][p_size])bot_row), 1, row_t, proc_below,
        //           DEFAULT_TAG, MPI_COMM_WORLD, &(reqs[3]));

        MPI_Rsend(*((double(*)[][p_size])top_row), 1, row_t, proc_above,
                  DEFAULT_TAG, MPI_COMM_WORLD);
        MPI_Rsend(*((double(*)[][p_size])bot_row), 1, row_t, proc_below,
                  DEFAULT_TAG, MPI_COMM_WORLD);
#endif

        // Compute the rest of the rows.
        // Starting at top row because the first row is always ignored
        // double(*inner_rows)[n_rows - 2][p_size] = (double(*)[n_rows - 2][p_size])(&((*local_problem)[1][0]));
        // double(*inner_rows_prev)[n_rows - 2][p_size] = (double(*)[n_rows - 2][p_size])(&((*local_problem_prev)[1][0]));
        perform_iteration(n_rows - 2, p_size, *((double(*)[][p_size])top_row),
                          *((double(*)[][p_size])top_row_prev));

        if (rank == debug_rank)
        {
            printf("\n\n");
            printf("[%d] [it %d]chunk after inner rows: \n", rank, iterations);
            array_2d_print(n_rows, p_size, local_problem, rank);
        }

        // Perform check for convergence
        // has_converged = (iterations >= 3);
        has_converged = matrix_has_converged(precision, n_rows - 2, p_size,
                                             *((double(*)[][p_size])top_row),
                                             *((double(*)[][p_size])top_row_prev));
        // if (has_converged)
        //     printf("[%d] Converged after %d iterations \n", rank, iterations);

        // Copy problem to other buffer for next iteration
        // Not including the first and last rows as they will be copied in by
        // the receive message
        // size_t inner_rows_size = sizeof(double) * (size_t)p_size * (size_t)(n_rows - 4);
        // memcpy(inner_rows_prev, inner_rows, inner_rows_size);
        // memcpy(first_row_prev, first_row, sizeof(double) * (size_t)p_size * (size_t)(n_rows));
        // memcpy(local_problem_prev, local_problem, sizeof(*local_problem_prev));
        memcpy(top_row_prev, top_row, sizeof(double) * (size_t)p_size * (size_t)(n_rows - 2));

        // sync_with_neighbours(rank, n_procs, n_rows,
        //                      p_size, *local_problem, row_t,
        //                      &has_converged, &has_top_converged,
        //                      &has_bot_converged, iterations);

        // Finish communicaiton with neighbours
#if ASYNC_COMMS
        MPI_Status statuses[request_count];
        MPI_Waitall(request_count, reqs, statuses);

        // Abort if any messages fail
        for (int i = 0; i < request_count; i++)
        {
            int error_code = statuses[i].MPI_ERROR;
            if (error_code != MPI_SUCCESS)
            {
                char err_str[MPI_MAX_ERROR_STRING] = "";
                int result_len;
                // MPI_Error_string(error_code, err_str, &result_len);
                fprintf(stderr,
                        "[%d] Aborting. Error sending message to neighbour: %s \n",
                        rank, err_str);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
#else
        MPI_Sendrecv(*((double(*)[][p_size])top_row), 1, row_t, proc_above,
                     DEFAULT_TAG, *((double(*)[][p_size])first_row_prev), 1,
                     row_t, proc_above, DEFAULT_TAG, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        MPI_Sendrecv(*((double(*)[][p_size])bot_row), 1, row_t, proc_below,
                     DEFAULT_TAG, *((double(*)[][p_size])last_row_prev), 1,
                     row_t, proc_below, DEFAULT_TAG, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
#endif

        // Check if all procs have converged
        bool all_have_converged = false;
        MPI_Allreduce(&has_converged, &all_have_converged, 1, MPI_BYTE, MPI_LAND, MPI_COMM_WORLD);

        // Forcing chunk to continue iterating until all nodes have converged
        has_converged = all_have_converged;
        iterations++;
    }

    printf("[%d] total iterations: %d \n", rank, iterations);

    // Ignore first row since otherwise it could overwrite another processe's
    // work during reconstruction.
    const double *sendbuf = &((*local_problem)[1][0]);
    MPI_Gatherv(sendbuf, row_counts[rank], row_t, problem_global, row_counts,
                recv_displs, row_t, ROOT, MPI_COMM_WORLD);

    // // Print final matrix for each node
    // for (int i = 0; i < n_procs; i++)
    // {
    //     if (rank == i)
    //     {
    //         printf("\n[%d] final chunk \n", rank);
    //         array_2d_print(n_rows, p_size, local_problem, rank);
    //     }
    //     MPI_Barrier(MPI_COMM_WORLD);
    // }

    // if (rank == ROOT)
    // {
    //     printf("\n\n");
    //     printf("Final array: \n");
    //     array_2d_print(p_size, p_size, problem_global, rank);
    // }

    // free(problem_global);
    free(local_problem);
    MPI_Type_free(&row_t);

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    MPI_Finalize();

    if (rank == ROOT)
    {
        printf("Solved in %fs \n", (end_time - start_time));
        test_result_matches_sync_impl(p_size, precision, *problem_global,
                                      &load_testcase_1);

        test_result_has_converged(p_size, problem_global, precision);
    }

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
            matrix[row][col] = 0.25 * neighbours_sum;
        }
    }
}

void send_to_neighbour(int rank, int n_procs, double *row, MPI_Datatype row_t,
                       bool has_converged, int iterations)
{
    // if (has_converged)
    // {
    //     printf("[%d] we have converged...telling above \n", rank);
    //     MPI_Send(&iterations, 1, MPI_INT, rank, CONVERGENCE_TAG,
    //              MPI_COMM_WORLD);
    // }
    MPI_Send(row, 1, row_t, rank, DEFAULT_TAG, MPI_COMM_WORLD);
}

void send_to_neighbours(int rank, int n_procs, double *send_top,
                        double *send_bot, MPI_Datatype row_t,
                        bool has_converged, bool has_top_converged, bool has_bot_converged,
                        int iterations)
{
    MPI_Request send_requests[2];
    MPI_Status send_statuses[2];

    // Send to neighbour above
    if ((rank - 1 >= 0) && !has_top_converged)
    {
        // if (has_converged)
        // {
        //     printf("[%d] we have converged...telling above \n", rank);
        //     MPI_Send(&iterations, 1, MPI_INT, rank - 1, CONVERGENCE_TAG,
        //              MPI_COMM_WORLD);
        // }
        MPI_Isend(send_top, 1, row_t, rank - 1, DEFAULT_TAG, MPI_COMM_WORLD,
                  &(send_requests[0]));
        // MPI_Send(send_top, 1, row_t, rank - 1, DEFAULT_TAG, MPI_COMM_WORLD);
    }

    // Send to neighbour below
    if ((rank + 1 < n_procs) && !has_bot_converged)
    {
        // if (has_converged)
        // {
        //     printf("[%d] we have converged...telling bellow \n", rank);
        //     MPI_Send(&iterations, 1, MPI_INT, rank + 1, CONVERGENCE_TAG,
        //              MPI_COMM_WORLD);
        // }
        MPI_Isend(send_bot, 1, row_t, rank + 1, DEFAULT_TAG, MPI_COMM_WORLD,
                  &(send_requests[1]));
    }

    MPI_Waitall(2, send_requests, send_statuses);

    for (int i = 0; i < 2; i++)
    {
        int err = send_statuses[i].MPI_ERROR;
        if (err != MPI_SUCCESS)
        {
            char err_str[MPI_MAX_ERROR_STRING];
            int resultlen;
            MPI_Error_string(err, err_str, &resultlen);
            fprintf(stderr,
                    "[%d] Aborting. Error sending message to neighbour: %s \n",
                    rank, err_str);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}

void receive_from_neighbours(int rank, int n_procs, double *recv_top,
                             double *recv_bot, MPI_Datatype row_t,
                             bool has_top_converged, bool has_bot_converged,
                             int *max_neighbour_iterations)
{
    MPI_Request recv_requests[2];
    MPI_Status recv_statuses[2];

    int expected_msg_count;
    if ((rank - 1 >= 0) && (rank + 1 < n_procs))
    {
        expected_msg_count = 2;
    }
    else
    {
        expected_msg_count = 1;
    }

    int msg_count = 0;
    while (msg_count < expected_msg_count)
    {
        int msg_exists = (int)false;
        MPI_Iprobe(rank - 1, DEFAULT_TAG, MPI_COMM_WORLD, &msg_exists, MPI_STATUS_IGNORE);
        if (msg_exists)
        {
            MPI_Recv(recv_top, 1, row_t, rank - 1, DEFAULT_TAG, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            msg_exists = (int)false;
            msg_count++;
        }

        MPI_Iprobe(rank + 1, DEFAULT_TAG, MPI_COMM_WORLD, &msg_exists, MPI_STATUS_IGNORE);
        if (msg_exists)
        {
            MPI_Recv(recv_bot, 1, row_t, rank + 1, DEFAULT_TAG, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            msg_count++;
        }
    }
    // Receive from neighbour above
    // if ((rank - 1 >= 0) && !has_top_converged)
    // {
    //     // MPI_Recv(recv_top, 1, row_t, rank - 1, DEFAULT_TAG, MPI_COMM_WORLD,
    //     //          MPI_STATUS_IGNORE);
    //     MPI_Irecv(recv_top, 1, row_t, rank - 1, DEFAULT_TAG, MPI_COMM_WORLD,
    //               &(recv_requests[0]));

    //     // check_proc_converged(rank, rank - 1, has_top_converged);
    // }

    // // Receive from neighbour below
    // if ((rank + 1 < n_procs) && !has_bot_converged)
    // {
    //     // MPI_Recv(recv_bot, 1, row_t, rank + 1, DEFAULT_TAG, MPI_COMM_WORLD,
    //     //          MPI_STATUS_IGNORE);
    //     MPI_Irecv(recv_bot, 1, row_t, rank + 1, DEFAULT_TAG, MPI_COMM_WORLD,
    //               &(recv_requests[1]));

    //     // check_proc_converged(rank, rank + 1, has_bot_converged);
    // }

    // MPI_Waitall(2, recv_requests, recv_statuses);
    // for (int i = 0; i < 2; i++)
    // {
    //     int err = recv_statuses[i].MPI_ERROR;
    //     if (err != MPI_SUCCESS)
    //     {
    //         char err_str[MPI_MAX_ERROR_STRING];
    //         int resultlen;
    //         MPI_Error_string(err, err_str, &resultlen);
    //         fprintf(stderr,
    //                 "[%d] Aborting. Error sending message to neighbour: %s \n",
    //                 rank, err_str);
    //         MPI_Abort(MPI_COMM_WORLD, 1);
    //     }
    // }
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
                          MPI_Datatype row_t, bool *has_converged,
                          bool *has_top_converged, bool *has_bot_converged,
                          int iterations)
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
    double *send_top_row = &(local_problem[1][0]);
    double *send_bot_row = &(local_problem[n_rows - 2][0]);
    double *recv_top_row = &(local_problem[0][0]);
    double *recv_bot_row = &(local_problem[n_rows - 1][0]);

    /*
    Choosing not to use MPI struct type.
    Cannot have pointer to start of row inside of the struct. MPIs API provides no way to derefence it as expected when communicating.
    So need a flat data structure with no nested pointers. Would iether need to copy the contents into a static array or add the flag as a new column to the matrix.
    This is because MPIs API only allows the passing of a pointer to the start of the message, then the rest of the message is expected to exist next to it in memory.

    So to send a struct, the flag and matrix row would need to be owned by the struct, effectively meaning you have ot copy the row into the struct or store the flag next to hte row.
    either works.

    Sending te flag each time with the row is actually unnecessary, since the process only cares when it is set to true and it will remain false until it is true then stay true.
    So in theory this only needs to be communicated once. This could be done with a non blocking send and checking if the message exists each time we want to sychronise.

    The previous approach isn't suboptimal soley because we're sending redundant information with each communication most of the time, but mostly because it leeds to an awkward memeory representation
    or additional copys of the data. Along with being synchronous overhead whihc is performed with each iteration, this would also limit the problem size slightly by wasting memory storing redundant copies
    of potentially 10s of MBs of data.

    I think I want to send a single message then filter the message queue (which should be short) each iteration for the convergence message. This could be filtered with a special message TAG.

    thi can be done by sending an extra message when it has converged and using immediate probe at the start of synching to check for convergece messages from its neighbours. (either 1 or two)
    The message is a convergece message if it has a special tag. Perhaps it would be best to send this after syncing as a way of communicating that no further requests should be made to this root.

    That way it also leads into a natural progression into re-balancing the load.*/

    int max_neighbour_iterations = 0;
    if (rank % 2 == 0)
    {
        // Use sendRecv
        // Perhaps replace these if statements with a top and bot neighbour and use MPI_NULL_PROCESS when they are invalid

        // send_to_neighbours(rank, n_procs, send_top_row, send_bot_row, row_t,
        //                    *has_converged, *has_top_converged, *has_bot_converged,
        //                    iterations);
        // receive_from_neighbours(rank, n_procs, recv_top_row, recv_bot_row, row_t,
        //                         *has_top_converged, *has_bot_converged,
        //                         &max_neighbour_iterations);
    }
    else
    {
        // receive_from_neighbours(rank, n_procs, recv_top_row, recv_bot_row, row_t,
        //                         *has_top_converged, *has_bot_converged,
        //                         &max_neighbour_iterations);
        // send_to_neighbours(rank, n_procs, send_top_row, send_bot_row, row_t,
        //                    *has_converged, *has_top_converged, *has_bot_converged,
        //                    iterations);
    }

    // printf("[%d] recv_top: ", rank);
    // array_2d_print(1, n_cols, (double(*)[1][n_cols])recv_top_row, rank);

    // if (rank == ROOT)
    // {
    //     printf("[%d] recv_bot: ", rank);
    //     array_2d_print(1, n_cols, (double(*)[1][n_cols])recv_bot_row, rank);
    // }

    // Check top and bottom for convergence.
    // if ((rank - 1 >= 0) && !(*has_top_converged))
    // {
    //     check_proc_converged(rank, rank - 1, has_top_converged);
    // }

    // if ((rank + 1 < n_procs) && !(*has_bot_converged))
    // {
    //     check_proc_converged(rank, rank + 1, has_bot_converged);
    // }

    bool proc_converged[n_procs];
    if (*has_converged)
    {
        printf("[%d] still converged \n", rank);
    }
    MPI_Allgather(has_converged, 1, MPI_C_BOOL, proc_converged, 1,
                  MPI_C_BOOL, MPI_COMM_WORLD);

    if (rank == ROOT)
    {
        printf("has converged: ");
        for (int i = 0; i < n_procs; i++)
        {
            printf("%s ", proc_converged[i] ? "true" : "false");
        }
        printf("\n");
    }

    // Forcing this chunk to cary on computations until all nodes hace converged
    for (int i = 0; i < n_procs; i++)
    {
        if (!proc_converged[i])
        {
            *has_converged = false;
            break;
        }
    }
}

void check_proc_converged(int rank, int source, bool *has_converged)
{
    int message_exists = 0;
    MPI_Iprobe(source, CONVERGENCE_TAG, MPI_COMM_WORLD, &message_exists,
               MPI_STATUS_IGNORE);

    if (message_exists)
    {
        printf("[%d] Convergence message found from [%d] \n", rank, source);
        MPI_Recv(has_converged, 1, MPI_C_BOOL, source, CONVERGENCE_TAG,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
            {
                is_converged = false;
                // printf("first mistake at [%d][%d] \n", row, col);
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
        printf("FAIL Solutions don't match within the given precision (%f) \n", precision);
    }

    free(sync_solution);
    return sols_match;
}

bool test_result_has_converged(int p_size, double (*mat)[p_size][p_size], double precision)
{
    double(*mat_copy)[p_size][p_size];
    array_2d_try_alloc((size_t)p_size, (size_t)p_size, &mat_copy);
    memcpy(mat_copy, mat, sizeof(*mat_copy));

    perform_iteration(p_size, p_size, *mat_copy, *mat);
    bool has_converged = matrix_has_converged(precision, p_size, p_size, *mat_copy, *mat);

    if (has_converged)
    {
        printf("PASS Final matrix has successfully converged with a precision of %f \n", precision);
    }
    else
    {
        printf("FAIL Final matrix has not converged with a precision of %f \n", precision);
    }

    free(mat_copy);
    return has_converged;
}

int solve_sync(int p_size, double (*matrix)[p_size][p_size], double precision)
{
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
        converged = matrix_has_converged(precision, p_size, p_size, *matrix, *prev_matrix);
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