/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <concepts>
#include <iomanip>
#include <iostream>
#include <span>
#include <type_traits>

// test headers
#include "common/compare.h"
#include "common/cpu/inputs.h"
#include "common/halton.h"

namespace test::cpu {

  template <typename T>
  concept DivResultType = requires(T t) {
    // C struct
    requires std::is_standard_layout_v<T>;

    // quot data member
    t.quot;
    requires std::integral<decltype(T::quot)>;

    // rem data member
    t.rem;
    requires std::integral<decltype(T::rem)>;

    // members type
    requires std::same_as<decltype(T::quot), decltype(T::rem)>;
  };

  template <typename T>
  struct detailed {
    constexpr detailed(T value) : value_{value} {
    }

    T value_;
  };

  template <std::integral T>
  std::ostream& operator<<(std::ostream& out, detailed<T> const& val) {
    out << val.value_;
    return out;
  }

  template <DivResultType T>
  std::ostream& operator<<(std::ostream& out, detailed<T> const& val) {
    out << '{' << val.value_.quot << ", " << val.value_.rem << '}';
    return out;
  }

  template <DivResultType ResultType,
            std::integral InputType,
            ResultType (*XtdFunc)(InputType, InputType),
            ResultType (*RefFunc)(InputType, InputType)>
  inline void validate_div(const Device& device) {
    std::span<const InputType> values = inputs().values<InputType>();
    unsigned int size = values.size();
    for (unsigned int t = 0; t < size; ++t) {
      // generate a low-discrepancy deterministic sequence over [0, size)×[0, size)
      auto [i, j] = halton<2>(t, size);
      InputType x = values[i];
      InputType y = values[j];
      // compute the result of the xtd function
      ResultType result = XtdFunc(x, y);
      // compute the reference
      ResultType reference = RefFunc(x, y);
      // log the comparison
      INFO("input: " << detailed(x) << ", " << detailed(y) << "\nxtd result: " << detailed(result)
                     << "\nreference:  " << detailed(reference) << '\n');
      // compare the result with the reference
      compare(result.quot, reference.quot);
      compare(result.rem, reference.rem);
    }
  }

}  // namespace test::cpu
