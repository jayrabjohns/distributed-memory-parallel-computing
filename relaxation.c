#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

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
    // Argument parsing
    int sum_until = 0;

    if (argc >= 2)
        sum_until = (int)atol(argv[1]);

    if (sum_until <= 1)
    {
        printf("Provide the maximum to sum until. (int > 1)");
        fflush(stdout);
        exit(1);
    }

    // Init MPI
    int rc;
    rc = MPI_Init(&argc, &argv);
    if (rc != MPI_SUCCESS)
    {
        printf("MPI_Init failed \n");
        return 1;
    }

    int num_procs;
    rc = MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    if (rc != MPI_SUCCESS)
    {
        printf("MPI_Comm_size failed \n");
        return 1;
    }
    int num_workers = num_procs - 1;

    int rank;
    rc = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rc != MPI_SUCCESS)
    {
        printf("MPI_Comm_rank failed \n");
        return 1;
    }

    // Communicate the sums upper bound
    rc = MPI_Bcast(&sum_until, 1, MPI_LONG, ROOT, MPI_COMM_WORLD);
    if (rc != MPI_SUCCESS)
    {
        printf("MPI_Bcast failed \n");
        return 1;
    }

    if (rank == ROOT)
    {
        // Block until all results are in
        printf("number of processes (including root): %d \n", num_procs);
        printf("summing 1 -> %d \n", sum_until);

        int total = 0;
        for (int process = 1; process <= num_workers; process++)
        {
            int res;
            MPI_Recv(&res, 1, MPI_INT, process, DEFAULT_TAG,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total = total + res;
        }

        printf("Result: %d \n", total);
        fflush(stdout);
    }
    else
    {
        // Calculate and send result to root
        int result = sum(rank, sum_until, num_workers);
        MPI_Send(&result, 1, MPI_INT, ROOT, DEFAULT_TAG, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
