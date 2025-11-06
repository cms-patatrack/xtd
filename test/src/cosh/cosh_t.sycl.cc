/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/cosh.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"
#include "mpfr_cosh.h"

constexpr int ulps_single = 4;
constexpr int ulps_double = 4;

TEST_CASE("xtd::cosh", "[cosh][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("float xtd::cosh(float)") {
            validate<float, float, xtd::cosh, mpfr_coshf>(platform, device, ulps_single);
          }

          SECTION("double xtd::cosh(double)") {
            validate<double, double, xtd::cosh, mpfr_cosh>(platform, device, ulps_double);
          }

          SECTION("double xtd::cosh(int)") {
            validate<double, int, xtd::cosh, mpfr_cosh>(platform, device, ulps_double);
          }

          SECTION("float xtd::coshf(float)") {
            validate<float, float, xtd::coshf, mpfr_coshf>(platform, device, ulps_single);
          }

          SECTION("float xtd::coshf(double)") {
            validate<float, double, xtd::coshf, mpfr_coshf>(platform, device, ulps_single);
          }

          SECTION("float xtd::coshf(int)") {
            validate<float, int, xtd::coshf, mpfr_coshf>(platform, device, ulps_single);
          }
        }
      }
    }
  }
}
