/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// C++
#include <cmath>
#include <cstdio>
#include <limits>

inline float reference_nanf(uint32_t arg) {
  constexpr int size = std::numeric_limits<uint32_t>::digits10 + 1;
  char buffer[size + 1] = {};
  std::snprintf(buffer, size, "%u", arg);
  return std::nanf(buffer);
}

inline double reference_nan(uint64_t arg) {
  constexpr int size = std::numeric_limits<uint64_t>::digits10 + 1;
  char buffer[size + 1] = {};
  std::snprintf(buffer, size, "%lu", arg);
  return std::nan(buffer);
}
