#!/bin/bash

executable=/gpfs/u/home/PCP6/PCP6hrkp/mpi_project/exp.xl
nranks="1 2 4 8 16 32 64 128 256 512 1024 2048"
npl="32 128 512 2048 8192"

for i in $nranks; do
    for j in $npl; do
        if [ $i -le $j ]; then
            srun --nodes=$(((i-1)/32+1)) --ntasks=$i --overcommit $executable $j &
        fi
    done
done
wait
