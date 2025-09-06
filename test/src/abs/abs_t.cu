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
#include "xtd/math/abs.h"

// test headers
#include "common/cuda_check.h"
#include "common/cuda_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::abs", "[abs][cuda]") {
  std::vector<double> values = generate_input_values();

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

      SECTION("float xtd::abs(float)") {
        test<float, float, xtd::abs, mpfr::fabs>(queue, values, ulps_float);
      }

      SECTION("double xtd::abs(double)") {
        test<double, double, xtd::abs, mpfr::fabs>(queue, values, ulps_double);
      }

      SECTION("int xtd::abs(int)") {
        test_i<int, xtd::abs, std::abs>(queue, values);
      }

      SECTION("long long xtd::abs(long long)") {
        test_i<long long, xtd::abs, std::abs>(queue, values);
      }

      CUDA_CHECK(cudaStreamDestroy(queue));
    }
  }
}
