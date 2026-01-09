/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/expm1.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"
#include "mpfr_expm1.h"

constexpr int ulps_single = 3;
constexpr int ulps_double = 3;

TEST_CASE("xtd::expm1", "[expm1][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("float xtd::expm1(float)") {
            validate<float, float, xtd::expm1, mpfr_expm1f>(platform, device, ulps_single);
          }

          SECTION("double xtd::expm1(double)") {
            validate<double, double, xtd::expm1, mpfr_expm1>(platform, device, ulps_double);
          }

          SECTION("double xtd::expm1(int)") {
            validate<double, int, xtd::expm1, mpfr_expm1>(platform, device, ulps_double);
          }

          SECTION("float xtd::expm1f(float)") {
            validate<float, float, xtd::expm1f, mpfr_expm1f>(platform, device, ulps_single);
          }

          SECTION("float xtd::expm1f(double)") {
            validate<float, double, xtd::expm1f, mpfr_expm1f>(platform, device, ulps_single);
          }

          SECTION("float xtd::expm1f(int)") {
            validate<float, int, xtd::expm1f, mpfr_expm1f>(platform, device, ulps_single);
          }
        }
      }
    }
  }
}
