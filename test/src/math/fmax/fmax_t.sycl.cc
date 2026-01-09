/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/fmax.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"
#include "mpfr_fmax.h"

constexpr int ulps_single = 2;  // 0 ULP according to the OpenCL documentation.
constexpr int ulps_double = 0;

TEST_CASE("xtd::fmax", "[fmax][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("float xtd::fmax(float, float)") {
            test::sycl::validate<float, float, xtd::fmax, mpfr_fmaxf>(platform, device, ulps_single);
          }

          SECTION("double xtd::fmax(double, double)") {
            test::sycl::validate<double, double, xtd::fmax, mpfr_fmax>(platform, device, ulps_double);
          }

          SECTION("double xtd::fmax(int, int)") {
            test::sycl::validate<double, int, xtd::fmax, mpfr_fmax>(platform, device, ulps_double);
          }

          SECTION("float xtd::fmaxf(float, float)") {
            test::sycl::validate<float, float, xtd::fmaxf, mpfr_fmaxf>(platform, device, ulps_single);
          }

          SECTION("float xtd::fmaxf(double, double)") {
            test::sycl::validate<float, double, xtd::fmaxf, mpfr_fmaxf>(platform, device, ulps_single);
          }

          SECTION("float xtd::fmaxf(int, int)") {
            test::sycl::validate<float, int, xtd::fmaxf, mpfr_fmaxf>(platform, device, ulps_single);
          }
        }
      }
    }
  }
}
