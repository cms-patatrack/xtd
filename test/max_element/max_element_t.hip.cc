
#include <algorithm>
#include <execution>
#include <numeric>
#include <ranges>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include "algorithm.h"

#include "common/hip_check.h"
#include <hip_runtime.h>
#include <rocthrust/copy.h>

TEST_CASE("max_elementHIP", "[max_element]") {
  const int N = 100;
  std::random_device rd;
  std::mt19937 rng(rd());

  std::vector<int> values(N);
  std::iota(values.begin(), values.end(), 0);
  std::shuffle(values.begin(), values.end(), rng);

  int* d_values;
  HIP_CHECK(hipMalloc(&d_values, N * sizeof(int)));
  HIP_CHECK(hipMemcpy(d_values, values.data(), N * sizeof(int), hipMemcpyHostToDevice));

  SECTION("Default comparison") {
    auto max_iter = xtd::max_element(d_values, d_values + N);
	int max;
	thrust::copy(thrust::hip::par, max_iter, max_iter + 1, &max);
    REQUIRE(max == N - 1);
  }

  SECTION("Greater comparison") {
    auto max_iter = xtd::max_element(d_values, d_values + N, std::greater<int>());
	int max;
	thrust::copy(thrust::hip::par, max_iter, max_iter + 1, &max);
    REQUIRE(max == 0);
  }

  HIP_CHECK(hipFree(d_values));
}
