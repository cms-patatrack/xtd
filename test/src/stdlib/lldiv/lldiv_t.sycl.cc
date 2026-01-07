/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/stdlib/lldiv.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate_div.h"
#include "reference_lldiv.h"

TEST_CASE("xtd::lldiv", "[fdim][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("lldiv_t xtd::lldiv(long long, long long)") {
            validate_div<lldiv_t, long long, xtd::lldiv, reference_lldiv>(platform, device);
          }
        }
      }
    }
  }
}
