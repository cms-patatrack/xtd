
#pragma once

#include "internal/defines.h"
#include <type_traits>

#if !defined(XTD_TARGET_CUDA) && !defined(XTD_TARGET_HIP) && !defined(XTD_TARGET_SYCL)
#include <cmath>
#endif

namespace xtd {

template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
XTD_DEVICE_FUNCTION inline constexpr T log(T x) {
#if defined(XTD_TARGET_CUDA)
  // CUDA device code
  return ::log(x);
#elif defined(XTD_TARGET_HIP)
  // HIP/ROCm device code
  return ::log(x);
#elif defined(XTD_TARGET_SYCL)
  // SYCL device code
  return sycl::log(x);
#else
  // standard C++ code
  return std::log(x);
#endif
}

XTD_DEVICE_FUNCTION inline constexpr float logf(float x) {
  return log(x);
}

XTD_DEVICE_FUNCTION inline constexpr long double logl(long double x) {
  return log(x);
}

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
XTD_DEVICE_FUNCTION inline constexpr double log(T x) {
	return log(static_cast<double>(x));
}

} // namespace xtd
