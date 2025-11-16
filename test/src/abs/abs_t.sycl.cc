/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// C++ standard headers
#include <cstdlib>

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/stdlib/abs.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"

TEST_CASE("xtd::abs", "[abs][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("float xtd::abs(float)") {
            validate<float, float, xtd::abs, std::abs>(platform, device);
          }

          SECTION("double xtd::abs(double)") {
            validate<double, double, xtd::abs, std::abs>(platform, device);
          }

          SECTION("int xtd::abs(int)") {
            validate<int, int, xtd::abs, std::abs>(platform, device);
          }

          SECTION("long xtd::abs(long)") {
            validate<long, long, xtd::abs, std::abs>(platform, device);
          }

          SECTION("long long xtd::abs(long long)") {
            validate<long long, long long, xtd::abs, std::abs>(platform, device);
          }
        }
      }
    }
  }
}
