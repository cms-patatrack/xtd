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

// test headers
#include "common/compare.h"
#include "common/cpu/inputs.h"
#include "common/halton.h"

namespace test::cpu {

  template <std::floating_point T>
  struct remquo_t {
    T rem;
    int quo;
  };

  template <typename T>
  struct detailed {
    constexpr detailed(T value) : value_{value} {
    }

    T value_;
  };

  template <std::floating_point T>
  std::ostream& operator<<(std::ostream& out, detailed<T> const& val) {
    std::ostringstream buffer;
    buffer << std::fixed << std::setprecision(std::numeric_limits<T>::max_digits10) << val.value_ << " ["
           << std::hexfloat << val.value_ << "]";
    out << buffer.str();
    return out;
  }

  template <std::integral T>
  std::ostream& operator<<(std::ostream& out, detailed<T> const& val) {
    out << val.value_;
    return out;
  }

  template <std::floating_point T>
  std::ostream& operator<<(std::ostream& out, detailed<remquo_t<T>> const& val) {
    std::ostringstream buffer;
    buffer << std::fixed << std::setprecision(std::numeric_limits<T>::max_digits10) << "{" << val.value_.rem << " ["
           << std::hexfloat << val.value_.rem << "], " << val.value_.quo << "}";
    out << buffer.str();
    return out;
  }

  template <std::floating_point ResultType,
            typename InputType,
            ResultType (*XtdFunc)(InputType, InputType, int*),
            ResultType (*RefFunc)(InputType, InputType, int*)>
  inline void validate_remquo(const Device& device, int ulps = 0) {
    std::span<const InputType> values = inputs().values<InputType>();
    unsigned int size = values.size();
    for (unsigned int t = 0; t < size; ++t) {
      // generate a low-discrepancy deterministic sequence over [0, size)×[0, size)
      auto [i, j] = halton<2>(t, size);
      InputType x = values[i];
      InputType y = values[j];
      // compute the result of the xtd function
      remquo_t<ResultType> result;
      result.rem = XtdFunc(x, y, &result.quo);
      // compute the reference
      remquo_t<ResultType> reference;
      reference.rem = RefFunc(x, y, &reference.quo);
      // log the comparison
      INFO("input: " << detailed(x) << ", " << detailed(y) << "\nxtd result: " << detailed(result)
                     << "\nreference:  " << detailed(reference) << '\n');
      // compare the result with the reference
      compare(result.rem, reference.rem, ulps);
      // the standard guarantees only the sign and the last three bits
      compare(result.quo % 8, reference.quo % 8);
    }
  }

}  // namespace test::cpu
