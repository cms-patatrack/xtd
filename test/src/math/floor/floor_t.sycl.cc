/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/floor.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"
#include "mpfr_floor.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::floor", "[floor][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("float xtd::floor(float)") {
            validate<float, float, xtd::floor, mpfr_floorf>(platform, device, ulps_single);
          }

          SECTION("double xtd::floor(double)") {
            validate<double, double, xtd::floor, mpfr_floor>(platform, device, ulps_double);
          }

          SECTION("double xtd::floor(int)") {
            validate<double, int, xtd::floor, mpfr_floor>(platform, device, ulps_double);
          }

          SECTION("float xtd::floorf(float)") {
            validate<float, float, xtd::floorf, mpfr_floorf>(platform, device, ulps_single);
          }

          SECTION("float xtd::floorf(double)") {
            validate<float, double, xtd::floorf, mpfr_floorf>(platform, device, ulps_single);
          }

          SECTION("float xtd::floorf(int)") {
            validate<float, int, xtd::floorf, mpfr_floorf>(platform, device, ulps_single);
          }
        }
      }
    }
  }
}
