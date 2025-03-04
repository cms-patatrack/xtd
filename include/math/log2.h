
#pragma once

#include "internal/defines.h"
#include <type_traits>

#if !defined(XTD_TARGET_CUDA) && !defined(XTD_TARGET_HIP) && !defined(XTD_TARGET_SYCL)
#include <cmath>
#endif

namespace xtd {

template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
XTD_DEVICE_FUNCTION inline constexpr T log2(T x) {
#if defined(XTD_TARGET_CUDA)
  // CUDA device code
  return ::log2(x);
#elif defined(XTD_TARGET_HIP)
  // HIP/ROCm device code
  return ::log2(x);
#elif defined(XTD_TARGET_SYCL)
  // SYCL device code
  return sycl::log2(x);
#else
  // standard C++ code
  return std::log2(x);
#endif
}

XTD_DEVICE_FUNCTION inline constexpr float log2f(float x) {
  return log2(x);
}

XTD_DEVICE_FUNCTION inline constexpr long double log2l(long double x) {
  return log2(x);
}

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
XTD_DEVICE_FUNCTION inline constexpr double log2(T x) {
	return log2(static_cast<double>(x));
}

} // namespace xtd
