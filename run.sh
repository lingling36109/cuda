#!/usr/bin/bash
#SBATCH -A m4341_g
#SBATCH -t 00:20:00
#SBATCH -C "gpu&hbm40g"
#SBATCH -N 1
#SBATCH -q regular

run_matrix() {
    echo
    echo =====================
    srun -G 1 -n 1 ./build/test_spmm matrices/$1/$1.mtx $2
    echo =====================
    echo
}

module load cudatoolkit/12.9

run_matrix nlpkkt120 64
run_matrix nlpkkt120 256

run_matrix delaunay_n24 64
run_matrix delaunay_n24 256

run_matrix Cube_Coup_dt0 64
run_matrix Cube_Coup_dt0 256

