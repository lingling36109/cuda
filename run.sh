#!/usr/bin/bash
#SBATCH -A m4341_g
#SBATCH -t 00:10:00
#SBATCH -C "gpu&hbm40g"
#SBATCH -N 1
#SBATCH -q regular


srun -G 1 -n 1 ./build/test_spmm matrices/nlpkkt120.mtx 64
srun -G 1 -n 1 ./build/test_spmm matrices/nlpkkt120.mtx 256

srun -G 1 -n 1 ./build/test_spmm matrices/delaunay_n24.mtx 64
srun -G 1 -n 1 ./build/test_spmm matrices/delaunay_n24.mtx 256

srun -G 1 -n 1 ./build/test_spmm matrices/Cube_Coup_dt0.mtx 64
srun -G 1 -n 1 ./build/test_spmm matrices/Cube_Coup_dt0.mtx 256
