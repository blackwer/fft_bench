#include <complex>
#include <cstring>
#include <random>

#include <benchmark/benchmark.h>

#ifdef FFT_BENCH_OMP
#include <omp.h>
#endif

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
#elif FFT_BENCH_DUCC
#include <ducc0/fft/fft.h>
#include <ducc0/infra/aligned_array.h>
// ugly hack, but it makes compilation easier
#include <ducc0/infra/threading.cc>
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
template <int N_per_dim, int dim>
static void run_fft(benchmark::State &state) {
    constexpr int N = std::pow(N_per_dim, dim);
    fftw_complex *in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex *out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);
    initialize_arrays(N, (double *)in, (double *)out);

    int n[dim];
    for (int i = 0; i < dim; ++i)
        n[i] = N_per_dim;

#ifdef FFT_BENCH_OMP
    int n_threads;
#pragma omp parallel
    n_threads = omp_get_num_threads();
    fftw_plan_with_nthreads(n_threads);
#endif
    fftw_plan p = fftw_plan_dft(dim, n, in, out, FFTW_FORWARD, FFTW_MEASURE);

    for (auto _ : state)
        fftw_execute(p);

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
}

#elif defined(FFT_BENCH_POCKET)
template <int N, int dim>
static void run_fft(benchmark::State &state) {
    static_assert(dim == 1, "Multiple dimensions not implemented for pocket");
    double *in = (double *)malloc(2 * sizeof(double) * N);
    double *out = (double *)malloc(2 * sizeof(double) * N);
    initialize_arrays(N, in, out);

    cfft_plan p = make_cfft_plan(N);
    for (auto _ : state)
        cfft_forward(p, in, 1.0);

    destroy_cfft_plan(p);
}
#elif defined(FFT_BENCH_KISS)
template <int N, int dim = 1>
static void run_fft(benchmark::State &state) {
    static_assert(dim == 1, "Multiple dimensions not implemented for KISS");
    kiss_fft_cpx *in = (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx) * N);
    kiss_fft_cpx *out = (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx) * N);
    initialize_arrays(N, (double *)in, (double *)out);
    kiss_fft_cfg p = kiss_fft_alloc(N, 0, NULL, NULL);

    for (auto _ : state)
        kiss_fft(p, in, out);

    kiss_fft_free(p);
}
#elif defined(FFT_BENCH_DUCC)
template <int N_per_dim, int dim>
static void run_fft(benchmark::State &state) {
    constexpr int N = std::pow(N_per_dim, dim);
    ducc0::fmav_info::shape_t shape, axes;

    for (size_t i = 0; i < dim; ++i) {
        shape.push_back(N_per_dim);
        axes.push_back(i);
    }
    ducc0::aligned_array<std::complex<double>> vin(N), vout(N);
    initialize_arrays(N, (double *)vin.data(), (double *)vout.data());
    ducc0::cfmav<std::complex<double>> in(vin.data(), shape);
    ducc0::vfmav<std::complex<double>> out(vout.data(), shape);

#ifdef FFT_BENCH_OMP
    size_t n_threads = omp_get_max_threads();
#else
    size_t n_threads = 1;
#endif

    for (auto _ : state)
        ducc0::c2c(in, out, axes, true, 1., n_threads);
}
#endif

BENCHMARK(run_fft<1 << 8, 1>);
BENCHMARK(run_fft<1 << 9, 1>);
BENCHMARK(run_fft<1 << 10, 1>);
BENCHMARK(run_fft<1 << 11, 1>);
BENCHMARK(run_fft<1 << 12, 1>);
BENCHMARK(run_fft<1 << 13, 1>);
BENCHMARK(run_fft<1 << 14, 1>);
BENCHMARK(run_fft<1 << 15, 1>);
BENCHMARK(run_fft<1 << 16, 1>);
BENCHMARK(run_fft<1 << 17, 1>);
BENCHMARK(run_fft<1 << 18, 1>);
BENCHMARK(run_fft<1 << 19, 1>);
BENCHMARK(run_fft<1 << 20, 1>);
BENCHMARK(run_fft<1 << 21, 1>);
BENCHMARK(run_fft<1 << 22, 1>);
BENCHMARK(run_fft<1 << 23, 1>);
BENCHMARK(run_fft<1 << 24, 1>);
BENCHMARK(run_fft<1 << 25, 1>);

#if defined(FFT_BENCH_MKL) | defined(FFT_BENCH_FFTW3) | defined(FFT_BENCH_DUCC)
BENCHMARK(run_fft<1 << 4, 2>);
BENCHMARK(run_fft<1 << 5, 2>);
BENCHMARK(run_fft<1 << 6, 2>);
BENCHMARK(run_fft<1 << 7, 2>);
BENCHMARK(run_fft<1 << 8, 2>);
BENCHMARK(run_fft<1 << 9, 2>);
BENCHMARK(run_fft<1 << 10, 2>);
BENCHMARK(run_fft<1 << 11, 2>);
BENCHMARK(run_fft<1 << 12, 2>);
BENCHMARK(run_fft<1 << 13, 2>);

BENCHMARK(run_fft<1 << 2, 3>);
BENCHMARK(run_fft<1 << 3, 3>);
BENCHMARK(run_fft<1 << 5, 3>);
BENCHMARK(run_fft<1 << 6, 3>);
BENCHMARK(run_fft<1 << 7, 3>);
BENCHMARK(run_fft<1 << 8, 3>);
BENCHMARK(run_fft<1 << 9, 3>);
#endif

BENCHMARK_MAIN();
