// C++ standard headers
#include <cmath>
#include <limits>
#include <vector>

// Catch2 headers
#define CATCH_CONFIG_MAIN
#include <catch.hpp>

// xtd headers
#include "math.h"

TEST_CASE("sinCPU", "[sin]") {
  auto const epsilon = std::numeric_limits<double>::epsilon();
  auto const epsilon_f = std::numeric_limits<float>::epsilon();

  std::vector<double> values{-1., 0., M_PI / 2, M_PI, 42.};

  for (auto &v : values) {
    REQUIRE_THAT(xtd::sin(static_cast<int>(v)),
                 Catch::Matchers::WithinAbs(std::sin(static_cast<int>(v)), epsilon));
    REQUIRE_THAT(xtd::sin(static_cast<float>(v)),
                 Catch::Matchers::WithinAbs(std::sin(v), epsilon_f));
    REQUIRE_THAT(xtd::sin(static_cast<double>(v)),
                 Catch::Matchers::WithinAbs(std::sin(v), epsilon));
    REQUIRE_THAT(xtd::sinf(static_cast<int>(v)),
                 Catch::Matchers::WithinAbs(sinf(static_cast<int>(v)), epsilon_f));
    REQUIRE_THAT(xtd::sinf(static_cast<float>(v)), Catch::Matchers::WithinAbs(sinf(v), epsilon_f));
    REQUIRE_THAT(xtd::sinf(static_cast<double>(v)), Catch::Matchers::WithinAbs(sinf(v), epsilon_f));
  }
}
