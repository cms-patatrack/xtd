/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Simone Balducci <simone.balducci@cern.ch>, Patrick Kopper <patrick.kopper@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <concepts>
#include <type_traits>

#include "xtd/internal/defines.h"
#include "xtd/internal/float16.h"

namespace xtd {

  template <typename T>
  concept Numeric = requires {
    std::is_arithmetic_v<T>;
    requires sizeof(T) <= 8;
  };

  template <typename T>
  concept floating_point = std::floating_point<std::remove_cvref_t<T>> ||
                           std::same_as<std::remove_cvref_t<T>, xtd::float16_t>;

}  // namespace xtd
