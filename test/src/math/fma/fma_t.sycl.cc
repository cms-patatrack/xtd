/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/fma.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"
#include "mpfr_fma.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::fma", "[fma][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("float xtd::fma(float, float, float)") {
            test::sycl::validate<float, float, xtd::fma, mpfr_fmaf>(platform, device, ulps_single);
          }

          SECTION("double xtd::fma(double, double, double)") {
            test::sycl::validate<double, double, xtd::fma, mpfr_fma>(platform, device, ulps_double);
          }

          SECTION("double xtd::fma(int, int, int)") {
            test::sycl::validate<double, int, xtd::fma, mpfr_fma>(platform, device, ulps_double);
          }

          SECTION("float xtd::fmaf(float, float, float)") {
            test::sycl::validate<float, float, xtd::fmaf, mpfr_fmaf>(platform, device, ulps_single);
          }

          SECTION("float xtd::fmaf(double, double, double)") {
            test::sycl::validate<float, double, xtd::fmaf, mpfr_fmaf>(platform, device, ulps_single);
          }

          SECTION("float xtd::fmaf(int, int, int)") {
            test::sycl::validate<float, int, xtd::fmaf, mpfr_fmaf>(platform, device, ulps_single);
          }
        }
      }
    }
  }
}
