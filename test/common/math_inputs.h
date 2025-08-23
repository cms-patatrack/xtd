/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <vector>
#include <cmath>
#include <numbers>
#include <limits>
#include <iostream>

// std::nextafter is constexpr only in C++23 and newer
#if __cplusplus >= 202302L
#define constexpr_cxx23 constexpr
#else
#define constexpr_cxx23
#endif

constexpr_cxx23 inline std::vector<double> generate_input_values() {
  std::vector<double> values;

  auto add_with_ulps = [&](double x) {
    values.push_back(x);

    // Double ULP shifts
    double up = x, down = x;
    for (int i = 0; i < 3; ++i) {
      up = std::nextafter(up, std::numeric_limits<double>::infinity());
      down = std::nextafter(down, -std::numeric_limits<double>::infinity());
      values.push_back(up);
      values.push_back(down);
    }

    // Float ULP shifts (converted back to double)
    float xf_up = static_cast<float>(x);
    float xf_down = xf_up;
    for (int i = 0; i < 3; ++i) {
      xf_up = std::nextafterf(xf_up, std::numeric_limits<float>::infinity());
      xf_down = std::nextafterf(xf_down, -std::numeric_limits<float>::infinity());
      values.push_back(static_cast<double>(xf_up));
      values.push_back(static_cast<double>(xf_down));
    }
  };

  // Integers
  for (int i = -20; i <= 20; ++i) {
    double val = static_cast<double>(i);
    add_with_ulps(val);
  }

  // Square roots
  for (int i = -20; i <= 20; ++i) {
    double val = std::sqrt(static_cast<double>(i));
    add_with_ulps(val);
    add_with_ulps(-val);
  }

  // Multiples of pi/4
  for (int i = -25; i <= 25; ++i) {
    double val = i * (std::numbers::pi / 4.);
    add_with_ulps(val);
  }

  // Multiples of pi/3
  for (int i = -19; i <= 19; ++i) {
    double val = i * (std::numbers::pi / 3.);
    add_with_ulps(val);
  }

  // Logarithms
  for (int i = 1; i <= 20; ++i) {
    double val = std::log(static_cast<double>(i));
    add_with_ulps(val);
    add_with_ulps(-val);
    val = std::log2(static_cast<double>(i));
    add_with_ulps(val);
    add_with_ulps(-val);
    val = std::log10(static_cast<double>(i));
    add_with_ulps(val);
    add_with_ulps(-val);
  }

  // Powers of e
  for (int i = -20; i <= 20; ++i) {
    double val = std::exp(static_cast<double>(i));
    add_with_ulps(val);
    add_with_ulps(-val);
  }

  // Powers of 2
  for (int i = -30; i <= 30; ++i) {
    double val = std::ldexp(1.0, i);
    add_with_ulps(val);
    add_with_ulps(-val);
  }

  // Powers of 10
  for (int i = -10; i <= 10; ++i) {
    double val = std::pow(10.0, i);
    add_with_ulps(val);
    add_with_ulps(-val);
  }

  return values;
}
