/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <concepts>

#include "xtd/math/fabs.h"

namespace xtd {

  /* Computes the absolute value of arg exactly, for signed integral types.
   */
  XTD_DEVICE_FUNCTION inline constexpr auto abs(std::signed_integral auto arg) { return (arg < 0) ? -arg : arg; }

  /* Computes the absolute value of arg, with the appropriate precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr auto abs(std::floating_point auto arg) { return xtd::fabs(arg); }

}  // namespace xtd
