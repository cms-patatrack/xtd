
#include <algorithm>
#include <execution>
#include <numeric>
#include <ranges>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include "algorithm.h"

TEST_CASE("sortCPU", "[sort]") {
  const int N = 100;
  std::random_device rd;
  std::mt19937 rng(rd());

  std::vector<int> values(N);
  std::iota(values.begin(), values.end(), 0);
  std::shuffle(values.begin(), values.end(), rng);

  SECTION("Default comparison") {
    xtd::sort(values.begin(), values.end());
    REQUIRE(std::ranges::equal(values, std::views::iota(0, N)));
  }

  SECTION("Greater comparison") {
    xtd::sort(values.begin(), values.end(), std::greater<int>());
    REQUIRE(std::ranges::equal(values, std::views::iota(0, N) | std::views::reverse));
  }

  SECTION("Unseq execution policy") {
    xtd::sort(std::execution::unseq, values.begin(), values.end());
    REQUIRE(std::ranges::equal(values, std::views::iota(0, N)));
  }

  SECTION("Unseq execution policy with greater comparison") {
    xtd::sort(std::execution::unseq, values.begin(), values.end(), std::greater<int>());
    REQUIRE(std::ranges::equal(values, std::views::iota(0, N) | std::views::reverse));
  }
}
