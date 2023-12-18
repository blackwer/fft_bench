#!/usr/bin/env python3

import json
import re

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('cairo')

mainfont = {
    'family': 'sans-serif',
    'color':  'black',
    'weight': 'normal',
    'size': 18,
}

arches = ['rome', 'skylake', 'icelake']
implementations = ['mkl', 'fftw3', 'mkl-omp', 'fftw3-omp', 'pocket', 'kiss', 'ducc', 'ducc-omp', 'sleef', 'sleef-omp']

cpu_data = {
    'rome': 'AMD EPYC 7742',
    'icelake': 'Intel Xeon Platinum 8362',
    'skylake': 'Intel Xeon Gold 6148',
}

def get_run_params(name: str):
    return eval(re.findall(r'\<.*?\>', name)[0].strip('<>'))

aggregate_data = {}
for arch in arches:
    aggregate_data[arch] = {}
    for implementation in implementations:
        with open(f'{implementation}-{arch}.json', 'r') as f:
            data = json.load(f)

        n_runs = len(data['benchmarks'])
        params = []
        for i, run in enumerate(data['benchmarks']):
            # N_per_dim, dim, timing
            params.append((*get_run_params(run['name']), run['real_time']))

        aggregate_data[arch][implementation] = params

def plot_st_dim(dim: int):
    for arch in arches:
        _, ax = plt.subplots(1, figsize=(12, 8))
        for impl, meas in aggregate_data[arch].items():
            if '-omp' in impl:
                continue
            params = list(zip(*filter(lambda param: param[1] == dim, meas)))
            if not params:
                continue
            sizes, _, timings = params
            if len(sizes):
                plt.loglog(sizes, timings, label=impl, linewidth=3)

        plt.title(f"{dim}D C2C on {cpu_data[arch]} (single-threaded)", fontdict=mainfont)
        plt.xlabel("FFT size", fontdict=mainfont)
        plt.ylabel("Time (µs)", fontdict=mainfont)
        ax.tick_params(labelsize=14, width=2)

        plt.legend(prop={'size':18})
        plt.savefig(f'{dim}d_c2c_st_{arch}.png', )

def plot_mt_dim(dim: int):
    for arch in arches:
        _, ax = plt.subplots(1, figsize=(12, 8))
        for impl, meas in aggregate_data[arch].items():
            if '-omp' not in impl:
                continue
            params = list(zip(*filter(lambda param: param[1] == dim, meas)))
            if not params:
                continue
            sizes, _, timings = params
            if len(sizes):
                plt.loglog(sizes, timings, label=impl, linewidth=3)
        plt.title(f"{dim}D C2C on {cpu_data[arch]} (multi-threaded)", fontdict=mainfont)
        plt.xlabel("FFT size", fontdict=mainfont)
        plt.ylabel("Time (µs)", fontdict=mainfont)
        ax.tick_params(labelsize=14, width=2)

        plt.legend(prop={'size':18})
        plt.savefig(f'{dim}d_c2c_mt_{arch}.png', )


for dim in range(1, 4):
    plot_st_dim(dim)
    plot_mt_dim(dim)
