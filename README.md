# Distributed memory parallel computing

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

3. Run the binary
```bash
./relaxation
```

## Running on a cluster
