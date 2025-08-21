
#include <algorithm>
#include <execution>
#include <numeric>
#include <ranges>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include "algorithm.h"

#include "common/cuda_check.h"
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

TEST_CASE("max_elementCUDA", "[max_element]") {
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
    auto max_iter = xtd::max_element(d_values, d_values + N);
    int max;
	thrust::copy(thrust::device, max_iter, max_iter+1, &max);
    REQUIRE(max == N - 1);
  }

  SECTION("Greater comparison") {
    auto max_iter = xtd::max_element(d_values, d_values + N, std::greater<int>());
    int max;
	thrust::copy(thrust::device, max_iter, max_iter+1, &max);
    REQUIRE(max == 0);
  }

  CUDA_CHECK(cudaFree(d_values));
}
