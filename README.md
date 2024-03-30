# Distributed parallel computing

Here I investigate the parallelisation of matrix relaxation over a distributed Azure cluster using MPI. 

MPI is the de facto standard for scalable message passing between parallel programs. This project uses the [OpenMPI](https://www.open-mpi.org/) implementation of this standard.

Matrix relaxation is sometimes also called the [Jacobi method](https://en.wikipedia.org/wiki/Relaxation_(iterative_method)). In essence, a sliding window is passed over a matrix to calculate the average of a cell with its neighbours. This is an interesting problem to parallelise because each iteration depends on the previous iteration, and the value of each cell depends on its neighbours. This is especially interesting when parallelising over a distributed system because communication costs over a network are much higher in comparison to communication over memory. You must carefully minimise the data being transferred between processes while maintaining correctness.

#### High level design
Broadly speaking:
1. The root process splits a large matrix into chunks and sends them to worker processes, which could be on the same machine or a different node.
2. After each iteration of relaxation, a check must be performed to see if the matrix has converged, or rather that its difference from the previous iteration is sufficiently small.
3. If it has not converged, processes communicate the boundary of their local problem to neighbouring chunks
4. Repeat 2-3
5. Once the problem has converged, the matrix is carefully reconstructed by the root process.

#### Asynchronous communications
This program carefully performs communications and computations simultaneously in a bid to reduce the communication overhead.

I've included an alternate implementation which uses synchronous communications in the [report](Investigation%20Report.pdf) as well as a comparison of performance between asynchronous and synchronous communications. It also discusses different communication strategies and the reasoning behind my specific choice of strategy.

#### Scalability investigation
The [report](Investigation%20Report.pdf) includes an investigation of the scalability of this system, including graphics. It provides calcuations of [Speedup](https://en.wikipedia.org/wiki/Speedup) & Efficiency as well as comments on [Amdahl's law](https://en.wikipedia.org/wiki/Amdahl's_law) and [Gustafson's law](https://en.wikipedia.org/wiki/Gustafson's_law).

#### Testing 
The [report](Investigation%20Report.pdf) also includes details on correctness testing.

## Running locally
1. Have OpenMPI installed
```bash
sudo apt install libopenmpi-dev
```
or build from source
https://docs.open-mpi.org/en/v5.0.x/installing-open-mpi/quickstart.html

2. Compile with mpicc
```bash
mpicc relaxation.c -o relaxation
```

3. Run with mpirun to spin up multiple nodes locally
```bash
# Usage: run.sh [num of nodes] [problem size] [precision]
./scripts/run.sh 4 20000 0.01
```

## Running on a cluster
This will look different depending on architecture. This project was run on an Azure cluster using Slurm as a workload manager.

1. ssh into the head node and compile as before.

2. Dispatch with slurm, it will look something like this:
```bash
#!/bin/bash
#SBATCH --account=<your account>
#SBATCH --partition=<your partition>
#SBATCH --job-name=<your job name>
#SBATCH --nodes=<number of nodes> 
#SBATCH --mail-type=END
#SBATCH --mail-user=<your email>
pwd
./relaxation
```

This project was tested on several node sizes and a number of MPI process on each node. If you're interested and would like more details on performance, take a look at the [report](Investigation%20Report.pdf).