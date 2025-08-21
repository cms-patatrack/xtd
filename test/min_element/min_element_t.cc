
#include <algorithm>
#include <execution>
#include <numeric>
#include <ranges>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include "algorithm.h"

TEST_CASE("min_elementCPU", "[min_element]") {
  const int N = 100;
  std::random_device rd;
  std::mt19937 rng(rd());

  std::vector<int> values(N);
  std::iota(values.begin(), values.end(), 0);
  std::shuffle(values.begin(), values.end(), rng);

  SECTION("Default comparison") {
    auto min = xtd::min_element(values.begin(), values.end());
    REQUIRE(*min == 0);
  }

  SECTION("Greater comparison") {
    auto min = xtd::min_element(values.begin(), values.end(), std::greater<int>());
     REQUIRE(*min == N - 1);
  }

  SECTION("Unseq execution policy") {
    auto min = xtd::min_element(std::execution::unseq, values.begin(), values.end());
    REQUIRE(*min == 0);
  }

  SECTION("Unseq execution policy with greater comparison") {
    auto min =
        xtd::min_element(std::execution::unseq, values.begin(), values.end(), std::greater<int>());
    REQUIRE(*min == N - 1);
  }
}
