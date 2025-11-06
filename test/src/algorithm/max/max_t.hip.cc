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
#include "common/hip/platform.h"
#include "common/hip/validate.h"
#include "byval_max.h"

TEST_CASE("xtd::max", "[max][hip]") {
  const auto& platform = test::hip::platform();
  DYNAMIC_SECTION("HIP platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("HIP device " << device.index() << ": " << device.name()) {
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
