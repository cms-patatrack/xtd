/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Simone Balducci <simone.balducci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// Basic operations
#include "math/abs.h"
#include "math/fabs.h"
#include "math/fmod.h"
#include "math/remainder.h"
//remquo
//fma
#include "math/fmax.h"
#include "math/fmin.h"
#include "math/fdim.h"
//nan

// Exponential functions
#include "math/exp.h"
#include "math/exp2.h"
#include "math/expm1.h"
#include "math/log.h"
#include "math/log10.h"
#include "math/log2.h"
#include "math/log1p.h"

// Power functions
#include "math/pow.h"
#include "math/sqrt.h"
#include "math/cbrt.h"
#include "math/hypot.h"
//hypot(x,y,z)

// Trigonometric functions
#include "math/sin.h"
#include "math/cos.h"
#include "math/tan.h"
#include "math/asin.h"
#include "math/acos.h"
#include "math/atan.h"
#include "math/atan2.h"

// Hyperbolic functions
#include "math/sinh.h"
#include "math/cosh.h"
#include "math/tanh.h"
#include "math/asinh.h"
#include "math/acosh.h"
#include "math/atanh.h"

// Error and gamma functions
//erf
//erfc
//tgamma
//lgamma

// Nearest integer floating point operations
#include "math/ceil.h"
#include "math/floor.h"
#include "math/trunc.h"
#include "math/round.h"
//lround
//llround
#include "math/nearbyint.h"
#include "math/rint.h"
//lrint
//llrint

// Floating point manipulation functions
//frexp
//ldexp
//modf
//scalbn
//scalbln
//ilogb
//logb
//nextafter
//nexttoward
//copysign

// Classification and comparison
#include "math/fpclassify.h"
#include "math/isfinite.h"
#include "math/isinf.h"
#include "math/isnan.h"
#include "math/isnormal.h"
//signbit
//isgreater
//isgreaterequal
//isless
//islessequal
//islessgreater
//isunordered
