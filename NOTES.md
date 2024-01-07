Through experimentation I've realised that the approach of chekcing for convergence individually in chunks and stopping computation doesn't work.

When reconstructed the solution isnt converged.

I imagine this to be even if a chunk has converged, if the neighbouring chunk iterates enough times then the error or difference between what the converged row is because its been kept constant and what it would have been if computation had continued. This error difference gets larger than the precision of the problem.

So eve if a chunk has converged locally we must check for convergence will all other nodes.

this essentially creates a barrier for nodes which is bad.

I suppose another way of doing things would be to instead send async messages and all nodes stop after all processes has produced a convergece message.

The negative of this is that it introduces an element of randomness. Nodes may complete iterations at different rates due to external factors. 
This could lead to the same issue as above. But would it? Possibly. 
Its essentially creating a race condition between the message that a process has converged and the processor performing work.


Might have to communicate between every iteration.


One possibility is to onyl start checking for convergence once the process itself has converged. And also send a convergence message with the number of iteration it took.

Instead of the described approach we will attempt an allgather.
An allgather works and is the simplest solution.

There doesn't seem to be an MPI_Iallreduce, so the allreduce has to stay as a blocking call, removing hte potential to do any work while waiting for the communication. Although, I'm spectical that even it did exist whether it would serve much benefit, since the message size is so small anyway (a byte) the overhead of additional rendezvous communications would surely outway the benefit of any overlapped work.







So the asynccommunication options is slower. I think this is because fo the rendezvous operations introducing extra latency, under the hodd they have to perform multiple communications to establish when the sender and receiver are ready. This is backed up by online research where it is apparently a common enough problem and commonly enough known that for halo exchanges (neighbour to neighbour communications) 2 sendrecvs for each dimension is usually faster than async communications.

I've come accross the halo exchange which seems to be perfect for modelling this problem.

Communicators communicate in a ring fashion since each row only requires to talk ot its neighbours


First implementation was to spit ranks into even and odd, even sends first while even receives first.
This is essentially a more synchronous version of sendrcv, so we have moved to that.

Could define Graph topology with MPI_DIST_GRAPH_CREATE and use it with MPI_neighbour_alltoallw or MPI_Ineighbour_alltoallw for the same communications in one MPI call,
but for the sake of conceptual simplicity I have left it as separate point to point communications. These are refered to as nighbourhood collectives or communicators.

```C
// After syncing hte message
    // Check top and bottom for convergence.
    if ((rank - 1 >= 0) && !(*has_top_converged))
    {
        check_proc_converged(rank, rank - 1, has_top_converged);
    }

    if ((rank + 1 < n_procs) && !(*has_bot_converged))
    {
        check_proc_converged(rank, rank + 1, has_bot_converged);
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


// When sending our row during synchronisation
        if (has_converged)
        {
            printf("[%d] we have converged...telling above \n", rank);
            MPI_Send(&has_converged, 1, MPI_C_BOOL, rank - 1, CONVERGENCE_TAG,
                     MPI_COMM_WORLD);
        }
```

