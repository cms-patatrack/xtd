// C++ standard headers
#include <cmath>
#include <limits>
#include <vector>

// CUDA headers
#include <cuda_runtime.h>

// Catch2 headers
#define CATCH_CONFIG_MAIN
#include <catch.hpp>

// xtd headers
#include "math.h"

// test headers
#include "common/cuda_check.h"

template <typename T>
__global__ void sinKernel(double *result, T input) {
  *result = static_cast<double>(xtd::sin(input));
}

template <typename T>
__global__ void sinfKernel(double *result, T input) {
  *result = static_cast<double>(xtd::sinf(input));
}

TEST_CASE("sinCUDA", "[sin]") {
  int deviceCount;
  cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);

  if (cudaStatus != cudaSuccess || deviceCount == 0) {
    std::cout << "No NVIDIA GPUs found, the test will be skipped." << std::endl;
    exit(EXIT_SUCCESS);
  }

  CUDA_CHECK(cudaSetDevice(0));
  cudaStream_t q;
  CUDA_CHECK(cudaStreamCreate(&q));

  std::vector<double> values{-1., 0., M_PI / 2, M_PI, 42.};

  double *result;
  int constexpr N = 6;
  CUDA_CHECK(cudaMallocAsync(&result, N * sizeof(double), q));

  for (auto v : values) {
    CUDA_CHECK(cudaMemsetAsync(result, 0x00, N * sizeof(double), q));

    sinKernel<<<1, 1, 0, q>>>(result + 0, static_cast<int>(v));
    CUDA_CHECK(cudaGetLastError());
    sinKernel<<<1, 1, 0, q>>>(result + 1, static_cast<float>(v));
    CUDA_CHECK(cudaGetLastError());
    sinKernel<<<1, 1, 0, q>>>(result + 2, static_cast<double>(v));
    CUDA_CHECK(cudaGetLastError());
    sinfKernel<<<1, 1, 0, q>>>(result + 3, static_cast<int>(v));
    CUDA_CHECK(cudaGetLastError());
    sinfKernel<<<1, 1, 0, q>>>(result + 4, static_cast<float>(v));
    CUDA_CHECK(cudaGetLastError());
    sinfKernel<<<1, 1, 0, q>>>(result + 5, static_cast<double>(v));
    CUDA_CHECK(cudaGetLastError());

    double resultHost[N];
    CUDA_CHECK(cudaMemcpyAsync(resultHost, result, N * sizeof(double), cudaMemcpyDeviceToHost, q));
    CUDA_CHECK(cudaStreamSynchronize(q));

    auto const epsilon = std::numeric_limits<double>::epsilon();
    auto const epsilon_f = std::numeric_limits<float>::epsilon();
    REQUIRE_THAT(resultHost[0], Catch::Matchers::WithinAbs(std::sin(static_cast<int>(v)), epsilon));
    REQUIRE_THAT(resultHost[1], Catch::Matchers::WithinAbs(std::sin(v), epsilon_f));
    REQUIRE_THAT(resultHost[2], Catch::Matchers::WithinAbs(std::sin(v), epsilon));
    REQUIRE_THAT(resultHost[3], Catch::Matchers::WithinAbs(sinf(static_cast<int>(v)), epsilon_f));
    REQUIRE_THAT(resultHost[4], Catch::Matchers::WithinAbs(sinf(v), epsilon_f));
    REQUIRE_THAT(resultHost[5], Catch::Matchers::WithinAbs(sinf(v), epsilon_f));
  }

  CUDA_CHECK(cudaFreeAsync(result, q));
  CUDA_CHECK(cudaStreamDestroy(q));
}
