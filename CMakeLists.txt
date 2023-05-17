cmake_minimum_required(VERSION 3.14)
project(
  fft_bench
  LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)
find_package(PkgConfig REQUIRED)
find_package(MKL REQUIRED)

pkg_check_modules(FFTW3 REQUIRED fftw3)

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

find_program(MAKE_EXECUTABLE NAMES make REQUIRED)

add_subdirectory(${PROJECT_SOURCE_DIR}/extern/benchmark)

add_executable(mkl_bench ${PROJECT_SOURCE_DIR}/src/fft_bench.cpp)
target_include_directories(mkl_bench PRIVATE ${FFT_BENCH_INCLUDES} ${MKL_H})
target_link_libraries(mkl_bench mkl_rt benchmark::benchmark)
target_compile_options(mkl_bench PRIVATE -march=native -DFFT_BENCH_MKL)

add_executable(fftw3_bench ${PROJECT_SOURCE_DIR}/src/fft_bench.cpp)
target_include_directories(fftw3_bench PRIVATE ${FFT_BENCH_INCLUDES} ${FFTW3_INCLUDE_DIRS})
target_link_libraries(fftw3_bench ${FFTW3_LIBRARIES}  benchmark::benchmark)
target_compile_options(fftw3_bench PRIVATE -march=native -DFFT_BENCH_FFTW3)

add_executable(pocket_bench ${PROJECT_SOURCE_DIR}/src/fft_bench.cpp ${PROJECT_SOURCE_DIR}/extern/pocketfft/pocketfft.c)
target_include_directories(pocket_bench PRIVATE ${FFT_BENCH_INCLUDES} ${PROJECT_SOURCE_DIR}/extern/pocketfft)
target_link_libraries(pocket_bench benchmark::benchmark)
target_compile_options(pocket_bench PRIVATE -march=native -DFFT_BENCH_POCKET)
