/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <concepts>

namespace xtd {

  /* The concept arithmetic<T> is satisfied if and only if T is an integral type or a floating point type.
   */
  template <class T>
  concept arithmetic = std::integral<T> or std::floating_point<T>;

}  // namespace xtd
