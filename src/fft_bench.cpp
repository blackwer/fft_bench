#include <complex>
#include <cstring>
#include <random>

#include <benchmark/benchmark.h>

#ifdef FFT_BENCH_MKL
#include <fftw/fftw3_mkl.h>
#elif FFT_BENCH_FFTW3
#include <fftw3.h>
#elif FFT_BENCH_POCKET
extern "C" {
#include <pocketfft.h>
}
#define FFTW_MEASURE 0
#elif FFT_BENCH_KISS
#include <kiss_fft.h>
#define FFTW_MEASURE 0
#endif

void initialize_arrays(int N, double *in, double *out) {
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    std::uniform_real_distribution<double> distr(-1.0, 1.0);

    for (int i = 0; i < 2 * N; ++i)
        in[i] = distr(generator);

    std::memset(out, 0, 2 * N * sizeof(double));
}

#if defined(FFT_BENCH_MKL) | defined(FFT_BENCH_FFTW3)
template <int N, int STRATEGY = FFTW_MEASURE>
static void run_1d_fft(benchmark::State &state) {
    fftw_complex *in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex *out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);
    initialize_arrays(N, (double *)in, (double *)out);
    fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, STRATEGY);

    for (auto _ : state)
        fftw_execute(p);

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
}

#elif defined(FFT_BENCH_POCKET)
template <int N, int STRATEGY = FFTW_MEASURE>
static void run_1d_fft(benchmark::State &state) {
    double *in = (double *)malloc(2 * sizeof(double) * N);
    double *out = (double *)malloc(2 * sizeof(double) * N);
    initialize_arrays(N, in, out);

    cfft_plan p = make_cfft_plan(N);
    for (auto _ : state)
        cfft_forward(p, in, 1.0);

    destroy_cfft_plan(p);
}
#elif defined(FFT_BENCH_KISS)
template <int N, int STRATEGY = FFTW_MEASURE>
static void run_1d_fft(benchmark::State &state) {
    kiss_fft_cpx *in = (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx) * N);
    kiss_fft_cpx *out = (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx) * N);
    initialize_arrays(N, (double *)in, (double *)out);
    kiss_fft_cfg p = kiss_fft_alloc(N, 0, NULL, NULL);

    for (auto _ : state)
        kiss_fft(p, in, out);

    kiss_fft_free(p);
}
#endif

BENCHMARK(run_1d_fft<1 << 8>);
BENCHMARK(run_1d_fft<1 << 9>);
BENCHMARK(run_1d_fft<1 << 10>);
BENCHMARK(run_1d_fft<1 << 11>);
BENCHMARK(run_1d_fft<1 << 12>);
BENCHMARK(run_1d_fft<1 << 13>);
BENCHMARK(run_1d_fft<1 << 14>);
BENCHMARK(run_1d_fft<1 << 15>);
BENCHMARK(run_1d_fft<1 << 16>);
BENCHMARK(run_1d_fft<1 << 17>);
BENCHMARK(run_1d_fft<1 << 18>);
BENCHMARK(run_1d_fft<1 << 19>);
BENCHMARK(run_1d_fft<1 << 20>);
BENCHMARK(run_1d_fft<1 << 21>);
BENCHMARK(run_1d_fft<1 << 22>);
BENCHMARK(run_1d_fft<1 << 23>);
BENCHMARK(run_1d_fft<1 << 24>);
BENCHMARK(run_1d_fft<1 << 25>);


BENCHMARK_MAIN();
