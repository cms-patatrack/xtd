/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// C++ standard headers
#include <algorithm>

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/algorithm/max.h"

// test headers
#include "common/cuda/platform.h"
#include "common/cuda/validate.h"
#include "byval_max.h"

TEST_CASE("xtd::max", "[max][cuda]") {
  const auto& platform = test::cuda::platform();
  DYNAMIC_SECTION("CUDA platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("CUDA device " << device.index() << ": " << device.name()) {
        SECTION("xtd::float16_t xtd::max(xtd::float16_t, xtd::float16_t)") {
          // NVCC suffers from internal template mangling errors on stateless lambdas
          // constexpr auto xtd::float16_t max = +[](xtd::float16_t a, xtd::float16_t b) -> xtd::float16_t { return xtd::float16_t(std::max(static_cast<float>(a), static_cast<float>(b))); };
          // validate<xtd::float16_t, xtd::float16_t, xtd::max, xtd::float16_t max>(device);
          validate<xtd::float16_t, xtd::float16_t, xtd::max, byval::max>(device);
        }

        SECTION("float xtd::max(float, float)") {
          validate<float, float, xtd::max, byval::max>(device);
        }

        SECTION("double xtd::max(double, double)") {
          validate<double, double, xtd::max, byval::max>(device);
        }

        SECTION("int xtd::max(int, int)") {
          validate<int, int, xtd::max, byval::max>(device);
        }

        SECTION("long xtd::max(long, long)") {
          validate<long, long, xtd::max, byval::max>(device);
        }

        SECTION("long long xtd::max(long long, long long)") {
          validate<long long, long long, xtd::max, byval::max>(device);
        }
      }
    }
  }
}
