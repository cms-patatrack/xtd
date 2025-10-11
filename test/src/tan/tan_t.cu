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

// CUDA headers
#include <cuda_runtime.h>

// mpfr::real headers
#include <real.hpp>

// xtd headers
#include "xtd/math/tan.h"

// test headers
#include "common/cuda_check.h"
#include "common/cuda_test.h"
#include "common/cuda_version.h"
#include "common/math_inputs.h"

constexpr int ulps_single = 4;
constexpr int ulps_double = 2;

constexpr auto ref_function = [](mpfr_double x) { return mpfr::tan(x); };
constexpr auto ref_functionf = [](mpfr_single x) { return mpfr::tan(x); };

TEST_CASE("xtd::tan", "[tan][cuda]") {
  std::vector<double> values = generate_input_values();

  DYNAMIC_SECTION("CUDA platform: " << cuda_version()) {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    for (int device = 0; device < deviceCount; ++device) {
      cudaDeviceProp properties;
      CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
      DYNAMIC_SECTION("CUDA device " << device << ": " << properties.name) {
        // set the current GPU
        CUDA_CHECK(cudaSetDevice(device));

        // create a CUDA stream for all the asynchronous operations on this GPU
        cudaStream_t queue;
        CUDA_CHECK(cudaStreamCreate(&queue));

        SECTION("float xtd::tan(float)") {
          test_a<float, float, xtd::tan, ref_function>(queue, values, ulps_single);
        }

        SECTION("double xtd::tan(double)") {
          test_a<double, double, xtd::tan, ref_function>(queue, values, ulps_double);
        }

        SECTION("double xtd::tan(int)") {
          test_a<double, int, xtd::tan, ref_function>(queue, values, ulps_double);
        }

        SECTION("float xtd::tanf(float)") {
          test_f<float, float, xtd::tanf, ref_functionf>(queue, values, ulps_single);
        }

        SECTION("float xtd::tanf(double)") {
          test_f<float, double, xtd::tanf, ref_functionf>(queue, values, ulps_single);
        }

        SECTION("float xtd::tanf(int)") {
          test_f<float, int, xtd::tanf, ref_functionf>(queue, values, ulps_single);
        }

        CUDA_CHECK(cudaStreamDestroy(queue));
      }
    }
  }
}
