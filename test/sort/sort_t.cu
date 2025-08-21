
#include <algorithm>
#include <numeric>
#include <ranges>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include "algorithm.h"

#include "common/cuda_check.h"
#include <cuda_runtime.h>

TEST_CASE("sortCUDA", "[sort]") {
  int deviceCount;
  cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);

  if (cudaStatus != cudaSuccess || deviceCount == 0) {
    std::cout << "No NVIDIA GPUs found, the test will be skipped." << std::endl;
    exit(EXIT_SUCCESS);
  }
  CUDA_CHECK(cudaSetDevice(0));

  const int N = 100;
  std::random_device rd;
  std::mt19937 rng(rd());

  std::vector<int> values(N);
  std::iota(values.begin(), values.end(), 0);
  std::shuffle(values.begin(), values.end(), rng);

  int* d_values;
  CUDA_CHECK(cudaMalloc(&d_values, N * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_values, values.data(), N * sizeof(int), cudaMemcpyHostToDevice));

  SECTION("Default comparison") {
    xtd::sort(d_values, d_values + N);
    CUDA_CHECK(cudaMemcpy(values.data(), d_values, N * sizeof(int), cudaMemcpyDeviceToHost));

    REQUIRE(std::ranges::equal(values, std::views::iota(0, N)));
  }

  SECTION("Greater comparison") {
    xtd::sort(d_values, d_values + N, std::greater<int>{});
    CUDA_CHECK(cudaMemcpy(values.data(), d_values, N * sizeof(int), cudaMemcpyDeviceToHost));

    REQUIRE(std::ranges::equal(values, std::views::iota(0, N) | std::views::reverse));
  }

  CUDA_CHECK(cudaFree(d_values));
}
