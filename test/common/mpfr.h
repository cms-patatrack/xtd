/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Patrick Kopper <patrick.kopper@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// MPFR headers
#include <mpfr.h>

// mpfr::real headers
#include <real.hpp>

// xtd headers
#include "xtd/internal/float16.h"

constexpr inline auto half_precision = 11;
using mpfr_half = mpfr::real<half_precision, MPFR_RNDN>;

constexpr inline auto single_precision = 24;
using mpfr_single = mpfr::real<single_precision, MPFR_RNDN>;

constexpr inline auto double_precision = 53;
using mpfr_double = mpfr::real<double_precision, MPFR_RNDN>;

// Correctly-rounded half-precision references.
//
// mpfr_half has the precision of a half-precision value, but MPFR's unbounded exponent range: when the
// exact result falls in the half-precision subnormal range, rounding it first to 11 bits and then to a
// subnormal value can give a result that is 1 ulp away from the correctly-rounded one. These helpers
// compute the result with the exponent range of IEEE 754 binary16 and round it once, also for subnormal
// results.
namespace mpfr_detail {

  // Exponent range of IEEE 754 binary16, in the MPFR convention (significand in [0.5, 1)).
  constexpr inline mpfr_exp_t half_emin = -23;
  constexpr inline mpfr_exp_t half_emax = 16;

  // Round the result of an MPFR function, computed with the binary16 exponent range, to binary16.
  inline xtd::float16_t round_to_half(mpfr_t result, int ternary) {
    ternary = mpfr_check_range(result, ternary, MPFR_RNDN);
    mpfr_subnormalize(result, ternary, MPFR_RNDN);
    // the result is a half-precision value, so its conversion to float is exact
    return xtd::float16_t(mpfr_get_flt(result, MPFR_RNDN));
  }

  // Set and restore the binary16 exponent range.
  class half_exponent_range {
  public:
    half_exponent_range() : emin_(mpfr_get_emin()), emax_(mpfr_get_emax()) {
      mpfr_set_emin(half_emin);
      mpfr_set_emax(half_emax);
    }
    ~half_exponent_range() {
      mpfr_set_emin(emin_);
      mpfr_set_emax(emax_);
    }

  private:
    mpfr_exp_t emin_;
    mpfr_exp_t emax_;
  };

}  // namespace mpfr_detail

// Correctly-rounded half-precision result of a univariate MPFR function, for example mpfr_sin.
inline xtd::float16_t mpfr_half_reference(int (*func)(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t), float x) {
  mpfr_t arg, result;
  mpfr_init2(arg, single_precision);
  mpfr_init2(result, half_precision);
  // like the half-precision function, round the argument to half precision first
  mpfr_set_flt(arg, static_cast<float>(xtd::float16_t(x)), MPFR_RNDN);
  xtd::float16_t value;
  {
    mpfr_detail::half_exponent_range range;
    value = mpfr_detail::round_to_half(result, func(result, arg, MPFR_RNDN));
  }
  mpfr_clears(arg, result, static_cast<mpfr_ptr>(nullptr));
  return value;
}

// Correctly-rounded half-precision result of a trivariate MPFR function, for example mpfr_fma.
inline xtd::float16_t mpfr_half_reference(int (*func)(mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t),
                                          float x,
                                          float y,
                                          float z) {
  mpfr_t arg_x, arg_y, arg_z, result;
  mpfr_inits2(single_precision, arg_x, arg_y, arg_z, static_cast<mpfr_ptr>(nullptr));
  mpfr_init2(result, half_precision);
  // like the half-precision function, round the arguments to half precision first
  mpfr_set_flt(arg_x, static_cast<float>(xtd::float16_t(x)), MPFR_RNDN);
  mpfr_set_flt(arg_y, static_cast<float>(xtd::float16_t(y)), MPFR_RNDN);
  mpfr_set_flt(arg_z, static_cast<float>(xtd::float16_t(z)), MPFR_RNDN);
  xtd::float16_t value;
  {
    mpfr_detail::half_exponent_range range;
    value = mpfr_detail::round_to_half(result, func(result, arg_x, arg_y, arg_z, MPFR_RNDN));
  }
  mpfr_clears(arg_x, arg_y, arg_z, result, static_cast<mpfr_ptr>(nullptr));
  return value;
}
