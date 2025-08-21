
#include <algorithm>
#include <numeric>
#include <ranges>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include "algorithm.h"

#include "common/hip_check.h"
#include <hip_runtime.h>

TEST_CASE("sortHIP", "[sort]") {
  int deviceCount;
  hipError_t hipStatus = hipGetDeviceCount(&deviceCount);

  if (hipStatus != hipSuccess || deviceCount == 0) {
    std::cout << "No AMD GPUs found, the test will be skipped." << std::endl;
    exit(EXIT_SUCCESS);
  }
  HIP_CHECK(hipSetDevice(0));

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
    xtd::sort(d_values, d_values + N);
    HIP_CHECK(hipMemcpy(values.data(), d_values, N * sizeof(int), hipMemcpyDeviceToHost));

    REQUIRE(std::ranges::equal(values, std::views::iota(0, N)));
  }

  SECTION("Greater comparison") {
    xtd::sort(d_values, d_values + N, std::greater<int>{});
    HIP_CHECK(hipMemcpy(values.data(), d_values, N * sizeof(int), hipMemcpyDeviceToHost));

    REQUIRE(std::ranges::equal(values, std::views::iota(0, N) | std::views::reverse));
  }

  HIP_CHECK(hipFree(d_values));
}
