
#include <algorithm>
#include <numeric>
#include <ranges>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include "algorithm.h"

#include <sycl/sycl.hpp>

TEST_CASE("sortSYCL", "[sort]") {
  const int N = 100;
  std::random_device rd;
  std::mt19937 rng(rd());

#ifdef ONEAPI_CPU
  auto queue = sycl::queue{sycl::cpu_selector_v, sycl::property::queue::in_order()};
#else
  if (sycl::device::get_devices(sycl::info::device_type::gpu).size() == 0) {
    std::cout << "No SYCL GPUs found, the test will be skipped." << std::endl;
    exit(EXIT_SUCCESS);
  }
  auto queue = sycl::queue{sycl::gpu_selector_v, sycl::property::queue::in_order()};
#endif

  std::vector<int> values(N);
  std::iota(values.begin(), values.end(), 0);
  std::shuffle(values.begin(), values.end(), rng);

  auto *d_values = sycl::malloc_device<int>(N, queue);
  queue.memcpy(d_values, values.data(), N * sizeof(int)).wait();

  SECTION("Default comparison") {
    auto min = xtd::min_element(d_values, d_values + N);
    REQUIRE(*min == 0);
  }

  SECTION("Greater comparison") {
    auto min = xtd::min_element(d_values, d_values + N, std::greater<int>{});
    REQUIRE(*min == N - 1);
  }

  sycl::free(d_values, queue);
}
