/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/nearbyint.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"
#include "mpfr_nearbyint.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::nearbyint", "[nearbyint][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("float xtd::nearbyint(float)") {
            validate<float, float, xtd::nearbyint, mpfr_nearbyintf>(platform, device, ulps_single);
          }

          SECTION("double xtd::nearbyint(double)") {
            validate<double, double, xtd::nearbyint, mpfr_nearbyint>(platform, device, ulps_double);
          }

          SECTION("double xtd::nearbyint(int)") {
            validate<double, int, xtd::nearbyint, mpfr_nearbyint>(platform, device, ulps_double);
          }

          SECTION("float xtd::nearbyintf(float)") {
            validate<float, float, xtd::nearbyintf, mpfr_nearbyintf>(platform, device, ulps_single);
          }

          SECTION("float xtd::nearbyintf(double)") {
            validate<float, double, xtd::nearbyintf, mpfr_nearbyintf>(platform, device, ulps_single);
          }

          SECTION("float xtd::nearbyintf(int)") {
            validate<float, int, xtd::nearbyintf, mpfr_nearbyintf>(platform, device, ulps_single);
          }
        }
      }
    }
  }
}
