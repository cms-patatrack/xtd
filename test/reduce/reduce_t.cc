
#include <algorithm>
#include <execution>
#include <numeric>
#include <ranges>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include "algorithm.h"

TEST_CASE("reduceCPU", "[reduce]") {
  const int N = 100;
  std::random_device rd;
  std::mt19937 rng(rd());

  std::vector<int> values(N);
  std::iota(values.begin(), values.end(), 0);

  SECTION("Default reduction") {
    auto red = xtd::reduce(values.begin(), values.end());
    REQUIRE(red == std::reduce(values.begin(), values.end()));
  }

  SECTION("Less comparison") {
    auto red = xtd::reduce(values.begin(), values.end(), -1, std::less<int>());
    REQUIRE(red == std::reduce(values.begin(), values.end(), -1, std::less<int>()));
  }

  SECTION("Unseq execution policy") {
    int red = xtd::reduce(std::execution::unseq, values.begin(), values.end());
    REQUIRE(red == std::reduce(values.begin(), values.end()));
  }

  SECTION("Unseq execution policy with less comparison") {
    auto red =
        xtd::reduce(std::execution::unseq, values.begin(), values.end(), -1, std::less<int>());
    REQUIRE(red ==
            std::reduce(std::execution::unseq, values.begin(), values.end(), -1, std::less<int>()));
  }

  SECTION("Reduction with initial value") {
    auto red = xtd::reduce(values.begin(), values.end(), 1);
    REQUIRE(red == std::reduce(values.begin(), values.end(), 1));
  }

  SECTION("Reduction with initial value and unseq policy") {
    auto red = xtd::reduce(std::execution::unseq, values.begin(), values.end(), 1);
    REQUIRE(red == std::reduce(std::execution::unseq, values.begin(), values.end(), 1));
  }
}
