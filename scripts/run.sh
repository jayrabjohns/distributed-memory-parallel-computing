#!/bin/bash
clear
./scripts/build.sh
mpirun -n "$1" relaxation "$2"