/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// C++ standard headers
#include <cmath>
#include <iostream>
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
#include "xtd/math/atan2.h"

// test headers
#include "common/hip_check.h"
#include "common/hip_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 3;
constexpr int ulps_double = 2;

constexpr auto ref_atan2 = [](mpfr_double y, mpfr_double x) { return mpfr::atan2(y, x); };
constexpr auto ref_atan2f = [](mpfr_single y, mpfr_single x) { return mpfr::atan2(y, x); };

TEST_CASE("xtd::atan2", "[atan2][hip]") {
  std::vector<double> values = generate_input_values();

  int deviceCount;
  hipError_t hipStatus = hipGetDeviceCount(&deviceCount);

  if (hipStatus != hipSuccess || deviceCount == 0) {
    std::cout << "No AMD GPUs found, the test will be skipped.\n\n";
    exit(EXIT_SUCCESS);
  }

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

      SECTION("float xtd::atan2(float, float)") {
        test_2<float, float, xtd::atan2, ref_atan2>(queue, values, ulps_float);
      }

      SECTION("double xtd::atan2(double, double)") {
        test_2<double, double, xtd::atan2, ref_atan2>(queue, values, ulps_double);
      }

      SECTION("double xtd::atan2(int, int)") {
        test_2<double, int, xtd::atan2, ref_atan2>(queue, values, ulps_double);
      }

      SECTION("float xtd::atan2f(float, float)") {
        test_2f<float, float, xtd::atan2f, ref_atan2f>(queue, values, ulps_float);
      }

      SECTION("float xtd::atan2f(double, double)") {
        test_2f<float, double, xtd::atan2f, ref_atan2f>(queue, values, ulps_float);
      }

      SECTION("float xtd::atan2f(int, int)") {
        test_2f<float, int, xtd::atan2f, ref_atan2f>(queue, values, ulps_float);
      }

      HIP_CHECK(hipStreamDestroy(queue));
    }
  }
}
