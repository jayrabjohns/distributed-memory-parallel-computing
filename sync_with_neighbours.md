```C
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
```