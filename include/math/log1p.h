
#pragma once

#include "internal/defines.h"
#include <type_traits>

#if !defined(XTD_TARGET_CUDA) && !defined(XTD_TARGET_HIP) && !defined(XTD_TARGET_SYCL)
#include <cmath>
#endif

namespace xtd {

template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
XTD_DEVICE_FUNCTION inline constexpr T log1p(T x) {
#if defined(XTD_TARGET_CUDA)
  // CUDA device code
  return ::log1p(x);
#elif defined(XTD_TARGET_HIP)
  // HIP/ROCm device code
  return ::log1p(x);
#elif defined(XTD_TARGET_SYCL)
  // SYCL device code
  return sycl::log1p(x);
#else
  // standard C++ code
  return std::log1p(x);
#endif
}

XTD_DEVICE_FUNCTION inline constexpr float log1pf(float x) {
  return std::log1p(x);
}

XTD_DEVICE_FUNCTION inline constexpr long double log1pl(long double x) {
  return std::log1p(x);
}

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
XTD_DEVICE_FUNCTION inline constexpr double log1p(T x) {
	return log1p(static_cast<double>(x));
}

} // namespace xtd
