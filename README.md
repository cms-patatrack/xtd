_For everything I long to do_  
_No matter when or where or who_  
_Has one thing in common, too_  
_It's a, it's a, it's a, it's a_ `sin()`  
  
by Chris Lowe and Neil Tennant

---

# xtd

C++ functions for CPU and GPUs with a consistent interface.


## Accuracy of mathematical functions

For a detailed comparison of the accuracy of different math library
implementations, see

  “Accuracy of Mathematical Functions in Single, Double, Double Extended, and
  Quadruple Precision”, Brian Gladman, Vincenzo Innocente, John Mather, Paul
  Zimmermann, hal-03141101v8, https://inria.hal.science/hal-03141101 .


### GNU libc

On a Linux system the math functions are usually provided by GNU libc.
The accuracy of the GNU libc math functions implementation is described in
https://www.gnu.org/software/libc/manual/html_node/Errors-in-Math-Functions.html
and summarised below for an x86_64 host system:

function         |      float |     double
-----------------|------------|------------
acosf / acos     |         1  |         1
acoshf / acosh   |         2  |         2
asinf / asin     |         1  |         1
asinhf / asinh   |         2  |         2
atan2f / atan2   |         2  |         1
atanf / atan     |         1  |         1
atanhf / atanh   |         2  |         2
cbrtf / cbrt     |         1  |         4
cosf / cos       |         1  |         1
coshf / cosh     |         2  |         2
erfcf / erfc     |         3  |         5
erff / erf       |         1  |         1
exp10f / exp10   |         1  |         2
exp2f / exp2     |         1  |         1
expf / exp       |         1  |         1
expm1f / expm1   |         1  |         1
fabsf / fabs     |         0  |         0 
fmaf / fma       |         0  |         0
fmodf / fmod     |         0  |         0
hypotf / hypot   |         1  |         1
j0f / j0         |         9  |         3
j1f / j1         |         9  |         4
jnf / jn         |         4  |         4
lgammaf / lgamma |         7  |         4
log10f / log10   |         2  |         2
log1pf / log1p   |         1  |         1
log2f / log2     |         1  |         2
logf / log       |         1  |         1
pow10f / pow10   |       n/a  |       n/a
powf / pow       |         1  |         1
sincosf / sincos |         1  |         1
sinf / sin       |         1  |         1
sinhf / sinh     |         2  |         2
sqrtf / sqrt     |         0  |         0
tanf / tan       |         1  |         1
tanhf / tanh     |         2  |         2
tgammaf / tgamma |         8  |         9
y0f / y0         |         9  |         3
y1f / y1         |         9  |         6
ynf / yn         |         3  |         3


### NVIDIA CUDA

The accuracy of the NVIDIA CUDA math functions implementation is described in
https://docs.nvidia.com/cuda/cuda-c-programming-guide/#standard-functions
and summarised below for GPUs with compute capability 5.2 or higher:

function         |      float |     double
-----------------|------------|------------
acosf / acos     |         2  |         2
acoshf / acosh   |         4  |         3
asinf / asin     |         2  |         2
asinhf / asinh   |         3  |         3
atan2f / atan2   |         3  |         2
atanf / atan     |         2  |         2
atanhf / atanh   |         3  |         2
cbrtf / cbrt     |         1  |         1
cosf / cos       |         2† |         2
coshf / cosh     |         2  |         1
erfcf / erfc     |         4  |         5
erff / erf       |         2  |         2
exp10f / exp10   |         2† |         1
exp2f / exp2     |         2  |         1
expf / exp       |         2† |         1
expm1f / expm1   |         1  |         1
fabsf / fabs     |         0  |         0 
fmaf / fma       |         0  |         0
fmodf / fmod     |         0  |         0
hypotf / hypot   |         3  |         2
j0f / j0         |         9* |         7ᕯ
j1f / j1         |         9* |         7ᕯ
jnf / jn         | 2 + 2.5×n* |       n/aᕯ
lgammaf / lgamma |         6' |         4"
log10f / log10   |         2† |         1
log1pf / log1p   |         1  |         1
log2f / log2     |         1† |         1
logf / log       |         1† |         1
pow10f / pow10   |       n/a  |       n/a
powf / pow       |         4† |         2
sincosf / sincos |         2† |         2
sinf / sin       |         2† |         2
sinhf / sinh     |         3  |         2
sqrtf / sqrt     |         0‡ |         0
tanf / tan       |         4† |         2
tanhf / tanh     |         2† |         1
tgammaf / tgamma |         5  |        10
y0f / y0         |         9* |         7ᕯ
y1f / y1         |         9* |         7ᕯ
ynf / yn         | 2 + 2.5×n* |       n/a

  - † unless compiled with `--use_fast_math`
  - ‡ unless compiled with `--use_fast_math` or `--prec-sqrt=false`
  - * for |x| < 8, otherwise, the maximum absolute error is 2.2×10⁻⁶
  - ᕯ for |x| < 8, otherwise, the maximum absolute error is 5×10⁻¹²
  - ' outside interval -10.001 … -2.264; larger inside
  - " outside interval -23.0001 … -2.2637; larger inside


### AMD HIP/ROCm

The accuracy of the AMD ROCm math functions implementation is described in
https://rocm.docs.amd.com/projects/HIP/en/latest/reference/math_api.html
and summarised below:

function         |      float |     double
-----------------|------------|------------
acosf / acos     |         1  |         1
acoshf / acosh   |         1  |         1
asinf / asin     |         2  |         1
asinhf / asinh   |         1  |         1
atan2f / atan2   |         3† |         2†
atanf / atan     |         2  |         1
atanhf / atanh   |         1  |         1
cbrtf / cbrt     |         2  |         1
cosf / cos       |         1  |         1
coshf / cosh     |         1  |         1
erfcf / erfc     |         2  |         2
erff / erf       |         4  |         4
exp10f / exp10   |         1  |         1
exp2f / exp2     |         1  |         1
expf / exp       |         1  |         1
expm1f / expm1   |         1  |         2†
fabsf / fabs     |         0  |         0 
fmaf / fma       |         0  |         0
fmodf / fmod     |         0  |         0
hypotf / hypot   |         1  |         1
j0f / j0         |       n/a  |       n/a
j1f / j1         |       n/a  |       n/a
jnf / jn         |       n/a  |       n/a
lgammaf / lgamma |         4  |         2
log10f / log10   |         2  |         1
log1pf / log1p   |         1  |         1
log2f / log2     |         1  |         1
logf / log       |         2  |         1
pow10f / pow10   |       n/a  |       n/a
powf / pow       |         1  |         1
sincosf / sincos |         1  |         1
sinf / sin       |         1  |         1
sinhf / sinh     |         1  |         1
sqrtf / sqrt     |         1  |         1
tanf / tan       |         2† |         2†
tanhf / tanh     |         2  |         1
tgammaf / tgamma |         6  |         6
y0f / y0         |       n/a  |       n/a
y1f / y1         |       n/a  |       n/a
ynf / yn         |       n/a  |       n/a

  - † 1 ULP for a reduced input range, according to the documentation

Note: in some cases the accuracy is documented only for a small range of values.


### Intel oneAPI

The SYCL specification does not describe the accuracy of the mathematical functions.
Since the SYCL programming model was originally designed as a high-level model for
the OpenCL API, the accuracy of the SYCL built-in functions is the same as the
corresponding OpenCL functions.

The accuracy of the OpenCL built-in functions is described in
https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_C.html#relative-error-as-ulps
and summarised below:

function         |      float |     double
-----------------|------------|------------
acosf / acos     |         4  |         4
acoshf / acosh   |         4  |         4
asinf / asin     |         4  |         4
asinhf / asinh   |         4  |         4
atan2f / atan2   |         6  |         6
atanf / atan     |         5  |         5*
atanhf / atanh   |         5  |         5
cbrtf / cbrt     |         2  |         2
cosf / cos       |         4  |         4 
coshf / cosh     |         4  |         4
erfcf / erfc     |        16  |        16
erff / erf       |        16  |        16
exp10f / exp10   |         3  |         3
exp2f / exp2     |         3  |         3
expf / exp       |         3  |         3
expm1f / expm1   |         3  |         3
fabsf / fabs     |         0  |         0 
fmaf / fma       |         0  |         0
fmaxf / fmax     |         0  |         0
fminf / fmin     |         0  |         0
fmodf / fmod     |         0  |         0
hypotf / hypot   |         4  |         4
j0f / j0         |       n/a  |       n/a
j1f / j1         |       n/a  |       n/a
jnf / jn         |       n/a  |       n/a
lgammaf / lgamma |       n/a  |       n/a
log10f / log10   |         3  |         3
log1pf / log1p   |         2  |         2
log2f / log2     |         3  |         3
logf / log       |         3  |         3
pow10f / pow10   |       n/a  |       n/a
powf / pow       |        16  |        16
sincosf / sincos |         4  |         4
sinf / sin       |         4  |         4 
sinhf / sinh     |         4  |         4
sqrtf / sqrt     |         3† |         0†
tanf / tan       |         5  |         5
tanhf / tanh     |         5  |         5
tgammaf / tgamma |        16  |        16
y0f / y0         |       n/a  |       n/a
y1f / y1         |       n/a  |       n/a
ynf / yn         |       n/a  |       n/a

  - † may be affected by the `-ffast-math` compiler option.

