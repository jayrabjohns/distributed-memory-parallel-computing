#!/bin/bash
clear
./scripts/build.sh
mpirun --oversubscribe -n "$1" relaxation "$2" "$3"