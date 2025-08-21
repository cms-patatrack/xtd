
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

TEST_CASE("min_elementHIP", "[min_element]") {
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
    auto min_iter = xtd::min_element(d_values, d_values + N);
	int min;
	thrust::copy(thrust::hip::par, min_iter, min_iter + 1, &min);
    REQUIRE(min == 0);
  }

  SECTION("Greater comparison") {
    auto min_iter = xtd::min_element(d_values, d_values + N, std::greater<int>());
	int min;
	thrust::copy(thrust::hip::par, min_iter, min_iter + 1, &min);
    REQUIRE(min == N - 1);
  }

  HIP_CHECK(hipFree(d_values));
}
