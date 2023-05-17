#include <cstring>
#include <complex>
#include <random>

#include <benchmark/benchmark.h>

#ifdef FFT_BENCH_MKL
#include <fftw/fftw3_mkl.h>
#elif FFT_BENCH_FFTW3
#include <fftw3.h>
#else
#endif

template <int N, int STRATEGY=FFTW_MEASURE>
static void run_1d_fft(benchmark::State &state) {
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    std::uniform_real_distribution<double> distr(-1.0, 1.0);

    fftw_complex *in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex *out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);

    for (int i = 0; i < N; ++i) {
        in[i][0] = distr(generator);
        in[i][1] = distr(generator);
    }
    std::memset(out, 0, N * sizeof(fftw_complex));

    fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, STRATEGY);

    for (auto _ : state)
        fftw_execute(p);

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
}

BENCHMARK(run_1d_fft<1024 << 0>);
BENCHMARK(run_1d_fft<1024 << 1>);
BENCHMARK(run_1d_fft<1024 << 2>);
BENCHMARK(run_1d_fft<1024 << 3>);
BENCHMARK(run_1d_fft<1024 << 4>);
BENCHMARK(run_1d_fft<1024 << 5>);
BENCHMARK(run_1d_fft<1024 << 6>);
BENCHMARK(run_1d_fft<1024 << 7>);

BENCHMARK_MAIN();
