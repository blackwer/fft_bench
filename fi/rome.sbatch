#!/bin/bash -e
#SBATCH --partition=scc
#SBATCH --nodes=1
#SBATCH --job-name=fft-bench
#SBATCH --output=rome.log
#SBATCH --constraint=rome

module purge
module use ~rblackwell/modules
module load gcc/12 fftw/3.3.10 intel-oneapi-mkl
module list

FFT_BENCH_PATH=$HOME/projects/codes/fft_bench/build-avx2

taskset -c 0 $FFT_BENCH_PATH/mkl_bench --benchmark_out=mkl-rome.json --benchmark_out_format=json --benchmark_time_unit=us
taskset -c 0 $FFT_BENCH_PATH/fftw3_bench --benchmark_out=fftw3-rome.json --benchmark_out_format=json --benchmark_time_unit=us
taskset -c 0 $FFT_BENCH_PATH/pocket_bench --benchmark_out=pocket-rome.json --benchmark_out_format=json --benchmark_time_unit=us
taskset -c 0 $FFT_BENCH_PATH/kiss_bench --benchmark_out=kiss-rome.json --benchmark_out_format=json --benchmark_time_unit=us
