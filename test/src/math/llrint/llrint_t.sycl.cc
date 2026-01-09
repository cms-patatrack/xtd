/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/llrint.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"
#include "mpfr_llrint.h"

TEST_CASE("xtd::llrint", "[llrint][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("long long xtd::llrint(float)") {
            validate<long long, float, xtd::llrint, mpfr_llrintf>(platform, device);
          }

          SECTION("long long xtd::llrint(double)") {
            validate<long long, double, xtd::llrint, mpfr_llrint>(platform, device);
          }

          SECTION("long long xtd::llrint(int)") {
            validate<long long, int, xtd::llrint, mpfr_llrint>(platform, device);
          }

          SECTION("long long xtd::llrintf(float)") {
            validate<long long, float, xtd::llrintf, mpfr_llrintf>(platform, device);
          }

          SECTION("long long xtd::llrintf(double)") {
            validate<long long, double, xtd::llrintf, mpfr_llrintf>(platform, device);
          }

          SECTION("long long xtd::llrintf(int)") {
            validate<long long, int, xtd::llrintf, mpfr_llrintf>(platform, device);
          }
        }
      }
    }
  }
}
