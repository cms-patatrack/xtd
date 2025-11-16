/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/sinh.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"
#include "mpfr_sinh.h"

constexpr int ulps_single = 4;
constexpr int ulps_double = 4;

TEST_CASE("xtd::sinh", "[sinh][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("float xtd::sinh(float)") {
            validate<float, float, xtd::sinh, mpfr_sinhf>(platform, device, ulps_single);
          }

          SECTION("double xtd::sinh(double)") {
            validate<double, double, xtd::sinh, mpfr_sinh>(platform, device, ulps_double);
          }

          SECTION("double xtd::sinh(int)") {
            validate<double, int, xtd::sinh, mpfr_sinh>(platform, device, ulps_double);
          }

          SECTION("float xtd::sinhf(float)") {
            validate<float, float, xtd::sinhf, mpfr_sinhf>(platform, device, ulps_single);
          }

          SECTION("float xtd::sinhf(double)") {
            validate<float, double, xtd::sinhf, mpfr_sinhf>(platform, device, ulps_single);
          }

          SECTION("float xtd::sinhf(int)") {
            validate<float, int, xtd::sinhf, mpfr_sinhf>(platform, device, ulps_single);
          }
        }
      }
    }
  }
}
