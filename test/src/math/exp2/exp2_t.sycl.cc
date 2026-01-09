/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/exp2.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"
#include "mpfr_exp2.h"

constexpr int ulps_single = 3;
constexpr int ulps_double = 3;

TEST_CASE("xtd::exp2", "[exp2][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("float xtd::exp2(float)") {
            validate<float, float, xtd::exp2, mpfr_exp2f>(platform, device, ulps_single);
          }

          SECTION("double xtd::exp2(double)") {
            validate<double, double, xtd::exp2, mpfr_exp2>(platform, device, ulps_double);
          }

          SECTION("double xtd::exp2(int)") {
            validate<double, int, xtd::exp2, mpfr_exp2>(platform, device, ulps_double);
          }

          SECTION("float xtd::exp2f(float)") {
            validate<float, float, xtd::exp2f, mpfr_exp2f>(platform, device, ulps_single);
          }

          SECTION("float xtd::exp2f(double)") {
            validate<float, double, xtd::exp2f, mpfr_exp2f>(platform, device, ulps_single);
          }

          SECTION("float xtd::exp2f(int)") {
            validate<float, int, xtd::exp2f, mpfr_exp2f>(platform, device, ulps_single);
          }
        }
      }
    }
  }
}
