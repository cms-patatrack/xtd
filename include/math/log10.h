
#pragma once

#include "internal/defines.h"
#include <type_traits>

#if !defined(XTD_TARGET_CUDA) && !defined(XTD_TARGET_HIP) && !defined(XTD_TARGET_SYCL)
#include <cmath>
#endif

namespace xtd {

template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
XTD_DEVICE_FUNCTION inline constexpr T log10(T x) {
#if defined(XTD_TARGET_CUDA)
  // CUDA device code
  return ::log10(x);
#elif defined(XTD_TARGET_HIP)
  // HIP/ROCm device code
  return ::log10(x);
#elif defined(XTD_TARGET_SYCL)
  // SYCL device code
  return sycl::log10(x);
#else
  // standard C++ code
  return std::log10(x);
#endif
}

XTD_DEVICE_FUNCTION inline constexpr float log10f(float x) {
  return log10(x);
}

XTD_DEVICE_FUNCTION inline constexpr long double log10l(long double x) {
  return log10(x);
}

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
XTD_DEVICE_FUNCTION inline constexpr double log10(T x) {
	return log10(static_cast<double>(x));
}

} // namespace xtd
