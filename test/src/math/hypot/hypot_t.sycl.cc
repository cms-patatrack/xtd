/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/hypot.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"
#include "mpfr_hypot.h"

constexpr int ulps_single = 4;
constexpr int ulps_double = 4;

TEST_CASE("xtd::hypot", "[hypot][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("float xtd::hypot(float, float)") {
            test::sycl::validate<float, float, xtd::hypot, mpfr_hypotf>(platform, device, ulps_single);
          }

          SECTION("double xtd::hypot(double, double)") {
            test::sycl::validate<double, double, xtd::hypot, mpfr_hypot>(platform, device, ulps_double);
          }

          SECTION("double xtd::hypot(int, int)") {
            test::sycl::validate<double, int, xtd::hypot, mpfr_hypot>(platform, device, ulps_double);
          }

          SECTION("float xtd::hypotf(float, float)") {
            test::sycl::validate<float, float, xtd::hypotf, mpfr_hypotf>(platform, device, ulps_single);
          }

          SECTION("float xtd::hypotf(double, double)") {
            test::sycl::validate<float, double, xtd::hypotf, mpfr_hypotf>(platform, device, ulps_single);
          }

          SECTION("float xtd::hypotf(int, int)") {
            test::sycl::validate<float, int, xtd::hypotf, mpfr_hypotf>(platform, device, ulps_single);
          }
        }
      }
    }
  }
}
