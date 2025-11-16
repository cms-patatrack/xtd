/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/cbrt.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"
#include "mpfr_cbrt.h"

constexpr int ulps_single = 2;
constexpr int ulps_double = 2;

TEST_CASE("xtd::cbrt", "[cbrt][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("float xtd::cbrt(float)") {
            validate<float, float, xtd::cbrt, mpfr_cbrtf>(platform, device, ulps_single);
          }

          SECTION("double xtd::cbrt(double)") {
            validate<double, double, xtd::cbrt, mpfr_cbrt>(platform, device, ulps_double);
          }

          SECTION("double xtd::cbrt(int)") {
            validate<double, int, xtd::cbrt, mpfr_cbrt>(platform, device, ulps_double);
          }

          SECTION("float xtd::cbrtf(float)") {
            validate<float, float, xtd::cbrtf, mpfr_cbrtf>(platform, device, ulps_single);
          }

          SECTION("float xtd::cbrtf(double)") {
            validate<float, double, xtd::cbrtf, mpfr_cbrtf>(platform, device, ulps_single);
          }

          SECTION("float xtd::cbrtf(int)") {
            validate<float, int, xtd::cbrtf, mpfr_cbrtf>(platform, device, ulps_single);
          }
        }
      }
    }
  }
}
