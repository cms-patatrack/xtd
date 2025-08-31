/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// C++ standard headers
#include <string>
#include <vector>
using namespace std::literals;

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// HIP headers
#include <hip/hip_runtime.h>

// mpfr::real headers
#include <real.hpp>

// xtd headers
#include "xtd/math/hypot.h"

// test headers
#include "common/hip_check.h"
#include "common/hip_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 3;
constexpr int ulps_double = 2;

constexpr auto ref_hypot = [](mpfr_double y, mpfr_double x) { return mpfr::hypot(y, x); };
constexpr auto ref_hypotf = [](mpfr_single y, mpfr_single x) { return mpfr::hypot(y, x); };

TEST_CASE("xtd::hypot", "[hypot][hip]") {
  std::vector<double> values = generate_input_values();

  int deviceCount;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));

  for (int device = 0; device < deviceCount; ++device) {
    hipDeviceProp_t properties;
    HIP_CHECK(hipGetDeviceProperties(&properties, device));
    std::string section = "HIP GPU "s + std::to_string(device) + ": "s + properties.name;
    SECTION(section) {
      // set the current GPU
      HIP_CHECK(hipSetDevice(device));

      // create a HIP stream for all the asynchronous operations on this GPU
      hipStream_t queue;
      HIP_CHECK(hipStreamCreate(&queue));

      SECTION("float xtd::hypot(float, float)") {
        test_2<float, float, xtd::hypot, ref_hypot>(queue, values, ulps_float);
      }

      SECTION("double xtd::hypot(double, double)") {
        test_2<double, double, xtd::hypot, ref_hypot>(queue, values, ulps_double);
      }

      SECTION("double xtd::hypot(int, int)") {
        test_2<double, int, xtd::hypot, ref_hypot>(queue, values, ulps_double);
      }

      SECTION("float xtd::hypotf(float, float)") {
        test_2f<float, float, xtd::hypotf, ref_hypotf>(queue, values, ulps_float);
      }

      SECTION("float xtd::hypotf(double, double)") {
        test_2f<float, double, xtd::hypotf, ref_hypotf>(queue, values, ulps_float);
      }

      SECTION("float xtd::hypotf(int, int)") {
        test_2f<float, int, xtd::hypotf, ref_hypotf>(queue, values, ulps_float);
      }

      HIP_CHECK(hipStreamDestroy(queue));
    }
  }
}
