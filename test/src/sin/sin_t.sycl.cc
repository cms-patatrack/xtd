/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/sin.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"
#include "mpfr_sin.h"

constexpr int ulps_single = 4;
constexpr int ulps_double = 4;

TEST_CASE("xtd::sin", "[sin][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("float xtd::sin(float)") {
            validate<float, float, xtd::sin, mpfr_sinf>(platform, device, ulps_single);
          }

          SECTION("double xtd::sin(double)") {
            validate<double, double, xtd::sin, mpfr_sin>(platform, device, ulps_double);
          }

          SECTION("double xtd::sin(int)") {
            validate<double, int, xtd::sin, mpfr_sin>(platform, device, ulps_double);
          }

          SECTION("float xtd::sinf(float)") {
            validate<float, float, xtd::sinf, mpfr_sinf>(platform, device, ulps_single);
          }

          SECTION("float xtd::sinf(double)") {
            validate<float, double, xtd::sinf, mpfr_sinf>(platform, device, ulps_single);
          }

          SECTION("float xtd::sinf(int)") {
            validate<float, int, xtd::sinf, mpfr_sinf>(platform, device, ulps_single);
          }
        }
      }
    }
  }
}
