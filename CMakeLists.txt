cmake_minimum_required(VERSION 3.14)
project(
  fft_bench
  LANGUAGES C CXX
)
set(CMAKE_CXX_STANDARD 17)

set (default_build_type "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message (STATUS "No build type specified. Setting build type to Release.")
  set (CMAKE_BUILD_TYPE "Release" CACHE STRING "Valid options: Debug, RelWithDebInfo, Release" FORCE)
endif()

set(BENCHMARK_ENABLE_TESTING off CACHE STRING "" FORCE)

set(
  FFT_BENCH_INCLUDES
  ${PROJECT_SOURCE_DIR}/include
)


find_package(PkgConfig REQUIRED)
find_package(MKL REQUIRED)
find_package(OpenMP REQUIRED)
pkg_check_modules(FFTW3 REQUIRED fftw3)

add_subdirectory(${PROJECT_SOURCE_DIR}/extern/benchmark)

set(KISSFFT_STATIC ON CACHE STRING "" FORCE)
set(KISSFFT_PKGCONFIG ON CACHE STRING "" FORCE)
set(KISSFFT_TOOLS OFF CACHE STRING "" FORCE)
set(KISSFFT_TEST OFF CACHE STRING "" FORCE)
set(KISSFFT_DATATYPE double CACHE STRING "" FORCE)
add_subdirectory(${PROJECT_SOURCE_DIR}/extern/kissfft)


add_executable(mkl_bench ${PROJECT_SOURCE_DIR}/src/fft_bench.cpp)
target_include_directories(mkl_bench PRIVATE ${FFT_BENCH_INCLUDES} ${MKL_H})
target_link_libraries(mkl_bench mkl_rt benchmark::benchmark)
target_compile_options(mkl_bench PRIVATE -march=native -DFFT_BENCH_MKL)

add_executable(mkl_omp_bench ${PROJECT_SOURCE_DIR}/src/fft_bench.cpp)
target_include_directories(mkl_omp_bench PRIVATE ${FFT_BENCH_INCLUDES} ${FFTW3_INCLUDE_DIRS})
target_link_libraries(mkl_omp_bench mkl_rt OpenMP::OpenMP_CXX benchmark::benchmark)
target_compile_options(mkl_omp_bench PRIVATE -march=native -DFFT_BENCH_MKL -DFFT_BENCH_OMP)

add_executable(fftw3_bench ${PROJECT_SOURCE_DIR}/src/fft_bench.cpp)
target_include_directories(fftw3_bench PRIVATE ${FFT_BENCH_INCLUDES} ${FFTW3_INCLUDE_DIRS})
target_link_libraries(fftw3_bench ${FFTW3_LIBRARIES} benchmark::benchmark)
target_compile_options(fftw3_bench PRIVATE -march=native -DFFT_BENCH_FFTW3)

add_executable(fftw3_omp_bench ${PROJECT_SOURCE_DIR}/src/fft_bench.cpp)
target_include_directories(fftw3_omp_bench PRIVATE ${FFT_BENCH_INCLUDES} ${FFTW3_INCLUDE_DIRS})
target_link_libraries(fftw3_omp_bench ${FFTW3_LIBRARIES} ${FFTW3_LIBDIR}/libfftw3_omp.so  OpenMP::OpenMP_CXX benchmark::benchmark)
target_compile_options(fftw3_omp_bench PRIVATE -march=native -DFFT_BENCH_FFTW3 -DFFT_BENCH_OMP)

add_executable(pocket_bench ${PROJECT_SOURCE_DIR}/src/fft_bench.cpp ${PROJECT_SOURCE_DIR}/extern/pocketfft/pocketfft.c)
target_include_directories(pocket_bench PRIVATE ${FFT_BENCH_INCLUDES} ${PROJECT_SOURCE_DIR}/extern/pocketfft)
target_link_libraries(pocket_bench benchmark::benchmark)
target_compile_options(pocket_bench PRIVATE -march=native -DFFT_BENCH_POCKET)

add_executable(kiss_bench ${PROJECT_SOURCE_DIR}/src/fft_bench.cpp)
target_include_directories(kiss_bench PRIVATE ${FFT_BENCH_INCLUDES})
target_link_libraries(kiss_bench benchmark::benchmark kissfft::kissfft)
target_compile_options(kiss_bench PRIVATE -march=native -DFFT_BENCH_KISS)
