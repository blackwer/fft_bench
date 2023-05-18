import json
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.use('cairo')

mainfont = {
    'family': 'sans-serif',
    'color':  'black',
    'weight': 'normal',
    'size': 18,
}

arches = ['rome', 'skylake', 'icelake']
implementations = ['mkl', 'fftw3', 'mkl-omp', 'fftw3-omp', 'pocket', 'kiss']

cpu_data = {
    'rome': 'AMD EPYC 7742',
    'icelake': 'Intel Xeon Platinum 8362',
    'skylake': 'Intel Xeon Gold 6148',
}

def get_run_size(name: str):
    return eval(re.findall(r'\<.*?\>', name)[0].strip('<>'))

aggregate_data = {}
for arch in arches:
    aggregate_data[arch] = {}
    for implementation in implementations:
        with open(f'{implementation}-{arch}.json', 'r') as f:
            data = json.load(f)

        n_runs = len(data['benchmarks'])
        sizes = np.empty(n_runs)
        timings = np.empty(n_runs)
        for i, run in enumerate(data['benchmarks']):
            sizes[i] = get_run_size(run['name'])
            timings[i] = run['real_time']

        aggregate_data[arch][implementation] = dict(
            sizes=sizes,
            timings=timings,
        )

        
for arch in arches:
    fig, ax = plt.subplots(1, figsize=(12, 8))
    for impl, meas in aggregate_data[arch].items():
        if '-omp' in impl:
            continue
        plt.loglog(meas['sizes'], meas['timings'], label=impl, linewidth=3)
    plt.title(f"1D C2C on {cpu_data[arch]} (single-threaded)", fontdict=mainfont)
    plt.xlabel("FFT size", fontdict=mainfont)
    plt.ylabel("Time (µs)", fontdict=mainfont)
    ax.tick_params(labelsize=14, width=2)

    plt.legend(prop={'size':18})
    plt.savefig(f'1d_c2c_st_{arch}.png', )


for arch in arches:
    fig, ax = plt.subplots(1, figsize=(12, 8))
    for impl, meas in aggregate_data[arch].items():
        if '-omp' not in impl:
            continue
        plt.loglog(meas['sizes'], meas['timings'], label=impl, linewidth=3)
    plt.title(f"1D C2C on {cpu_data[arch]} (multi-threaded)", fontdict=mainfont)
    plt.xlabel("FFT size", fontdict=mainfont)
    plt.ylabel("Time (µs)", fontdict=mainfont)
    ax.tick_params(labelsize=14, width=2)

    plt.legend(prop={'size':18})
    plt.savefig(f'1d_c2c_mt_{arch}.png', )
