/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
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
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"
#include "byval_max.h"

TEST_CASE("xtd::max", "[max][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("float xtd::max(float, float)") {
            validate<float, float, xtd::max, byval::max>(platform, device);
          }

          SECTION("double xtd::max(double, double)") {
            validate<double, double, xtd::max, byval::max>(platform, device);
          }

          SECTION("int xtd::max(int, int)") {
            validate<int, int, xtd::max, byval::max>(platform, device);
          }

          SECTION("long xtd::max(long, long)") {
            validate<long, long, xtd::max, byval::max>(platform, device);
          }

          SECTION("long long xtd::max(long long, long long)") {
            validate<long long, long long, xtd::max, byval::max>(platform, device);
          }
        }
      }
    }
  }
}
