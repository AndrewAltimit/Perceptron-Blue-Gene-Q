#!/bin/bash

executable=/home/parallel/2017/PPChorakp/mpi_project/exp.out
nranks="1 2 4 8"
npl="32 128 512 2048"

for i in $nranks; do
    for j in $npl; do
        if [ $i -le $j ]; then
            mpirun -np $i $executable $j
        fi
    done
done
