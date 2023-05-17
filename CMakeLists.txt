cmake_minimum_required(VERSION 3.14)
project(
  fft_bench
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)

find_package(MKL REQUIRED)

set (default_build_type "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message (STATUS "No build type specified. Setting build type to Release.")
  set (CMAKE_BUILD_TYPE "Release" CACHE STRING "Valid options: Debug, RelWithDebInfo, Release" FORCE)
endif()

set(
  FFT_BENCH_INCLUDES
  ${PROJECT_SOURCE_DIR}/include
)

find_program(MAKE_EXECUTABLE NAMES make REQUIRED)

include_directories(
  ${PROJECT_SOURCE_DIR}/extern/pocketfft
)

add_executable(mkl_bench src/mkl_bench.cpp)
target_include_directories(mkl_bench PRIVATE ${FFT_BENCH_INCLUDES})
target_link_libraries(mkl_bench mkl_rt)
target_compile_options(mkl_bench PRIVATE -march=native)
target_include_directories(mkl_bench PRIVATE ${MKL_H})
