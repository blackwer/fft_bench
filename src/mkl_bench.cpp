#include <cstdlib>
#include <fftw/fftw3_mkl.h>

int main(int argc, char *argv[]) {
    int N = 1024;
    if (argc > 1)
        N = atof(argv[1]);

    fftw_complex *in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex *out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);
    fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_execute(p);

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
    return 0;
}
