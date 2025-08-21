
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

TEST_CASE("reduceHIP", "[reduce]") {
  const int N = 100;
  std::random_device rd;
  std::mt19937 rng(rd());

  std::vector<int> values(N);
  std::iota(values.begin(), values.end(), 0);
  std::shuffle(values.begin(), values.end(), rng);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  int* d_values;
  HIP_CHECK(hipMallocAsync(&d_values, N * sizeof(int), stream));
  HIP_CHECK(
      hipMemcpyAsync(d_values, values.data(), N * sizeof(int), hipMemcpyHostToDevice, stream));

  SECTION("Default reduction") {
    auto red = xtd::reduce(d_values, d_values + N);
    REQUIRE(red == std::reduce(values.begin(), values.end()));
  }

  SECTION("Less comparison") {
    auto red = xtd::reduce(d_values, d_values + N, -1, std::less<int>());
    REQUIRE(red == std::reduce(values.begin(), values.end(), -1, std::less<int>()));
  }

  SECTION("Reduction with initial value") {
    auto red = xtd::reduce(d_values, d_values + N, 1);
    REQUIRE(red == std::reduce(values.begin(), values.end(), 1));
  }
}
