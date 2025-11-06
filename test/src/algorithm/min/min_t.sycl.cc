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
#include "xtd/algorithm/min.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"
#include "byval_min.h"

TEST_CASE("xtd::min", "[min][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("float xtd::min(float, float)") {
            validate<float, float, xtd::min, byval::min>(platform, device);
          }

          SECTION("double xtd::min(double, double)") {
            validate<double, double, xtd::min, byval::min>(platform, device);
          }

          SECTION("int xtd::min(int, int)") {
            validate<int, int, xtd::min, byval::min>(platform, device);
          }

          SECTION("long xtd::min(long, long)") {
            validate<long, long, xtd::min, byval::min>(platform, device);
          }

          SECTION("long long xtd::min(long long, long long)") {
            validate<long long, long long, xtd::min, byval::min>(platform, device);
          }
        }
      }
    }
  }
}
