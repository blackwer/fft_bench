#!/bin/bash -e
#SBATCH --partition=scc
#SBATCH --nodes=1
#SBATCH --job-name=fft-bench
#SBATCH --output=icelake.log
#SBATCH --constraint=icelake

module purge
module use ~rblackwell/modules
module load gcc/12 fftw/3.3.10-avx512 intel-oneapi-mkl
module list

FFT_BENCH_PATH=$HOME/projects/codes/fft_bench/build

export OMP_NUM_THREADS=32
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
taskset -c 0-31 $FFT_BENCH_PATH/mkl_omp_bench --benchmark_out=mkl-omp-icelake.json --benchmark_out_format=json --benchmark_time_unit=us
taskset -c 0-31 $FFT_BENCH_PATH/fftw3_omp_bench --benchmark_out=fftw3-omp-icelake.json --benchmark_out_format=json --benchmark_time_unit=us
OMP_PROC_BIND= OMP_PLACES= OMP_WAIT_POLICY=passive taskset -c 0-31 $FFT_BENCH_PATH/ducc_omp_bench --benchmark_out=ducc-omp-icelake.json --benchmark_out_format=json --benchmark_time_unit=us
taskset -c 0-31 $FFT_BENCH_PATH/sleef_bench --benchmark_out=sleef-omp-icelake.json --benchmark_out_format=json --benchmark_time_unit=us

export OMP_NUM_THREADS=1
taskset -c 0 $FFT_BENCH_PATH/mkl_bench --benchmark_out=mkl-icelake.json --benchmark_out_format=json --benchmark_time_unit=us
taskset -c 0 $FFT_BENCH_PATH/fftw3_bench --benchmark_out=fftw3-icelake.json --benchmark_out_format=json --benchmark_time_unit=us
taskset -c 0 $FFT_BENCH_PATH/pocket_bench --benchmark_out=pocket-icelake.json --benchmark_out_format=json --benchmark_time_unit=us
taskset -c 0 $FFT_BENCH_PATH/kiss_bench --benchmark_out=kiss-icelake.json --benchmark_out_format=json --benchmark_time_unit=us
taskset -c 0 $FFT_BENCH_PATH/ducc_bench --benchmark_out=ducc-icelake.json --benchmark_out_format=json --benchmark_time_unit=us
taskset -c 0 $FFT_BENCH_PATH/sleef_bench --benchmark_out=sleef-icelake.json --benchmark_out_format=json --benchmark_time_unit=us
