/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// mpfr::real headers
#include <real.hpp>

constexpr inline auto single_precision = 24;
using mpfr_single = mpfr::real<single_precision, MPFR_RNDN>;

constexpr inline auto double_precision = 53;
using mpfr_double = mpfr::real<double_precision, MPFR_RNDN>;
