/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/asinh.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"
#include "mpfr_asinh.h"

constexpr int ulps_single = 4;
constexpr int ulps_double = 4;

TEST_CASE("xtd::asinh", "[asinh][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("float xtd::asinh(float)") {
            validate<float, float, xtd::asinh, mpfr_asinhf>(platform, device, ulps_single);
          }

          SECTION("double xtd::asinh(double)") {
            validate<double, double, xtd::asinh, mpfr_asinh>(platform, device, ulps_double);
          }

          SECTION("double xtd::asinh(int)") {
            validate<double, int, xtd::asinh, mpfr_asinh>(platform, device, ulps_double);
          }

          SECTION("float xtd::asinhf(float)") {
            validate<float, float, xtd::asinhf, mpfr_asinhf>(platform, device, ulps_single);
          }

          SECTION("float xtd::asinhf(double)") {
            validate<float, double, xtd::asinhf, mpfr_asinhf>(platform, device, ulps_single);
          }

          SECTION("float xtd::asinhf(int)") {
            validate<float, int, xtd::asinhf, mpfr_asinhf>(platform, device, ulps_single);
          }
        }
      }
    }
  }
}
