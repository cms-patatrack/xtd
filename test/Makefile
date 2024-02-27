# gcc
CXX := g++
GCC_TOOLCHAIN := $(abspath $(dir $(shell which $(CXX)))/..)
GCC_TARGET    := $(shell $(CXX) -dumpmachine)
HOST_CXXFLAGS := -O2 -fPIC -fdiagnostics-show-option -felide-constructors -fmessage-length=0 -fno-math-errno -ftree-vectorize -fvisibility-inlines-hidden --param vect-max-version-for-alias-checks=50 -msse3 -pipe -pthread -Werror=address -Wall -Werror=array-bounds -Wno-attributes -Werror=conversion-null -Werror=delete-non-virtual-dtor -Wno-deprecated -Werror=format-contains-nul -Werror=format -Wno-long-long -Werror=main -Werror=missing-braces -Werror=narrowing -Wno-non-template-friend -Wnon-virtual-dtor -Werror=overflow -Werror=overlength-strings -Wparentheses -Werror=pointer-arith -Wno-psabi -Werror=reorder -Werror=return-local-addr -Wreturn-type -Werror=return-type -Werror=sign-compare -Werror=strict-aliasing -Wstrict-overflow -Werror=switch -Werror=type-limits -Wunused -Werror=unused-but-set-variable -Wno-unused-local-typedefs -Werror=unused-value -Wno-error=unused-variable -Wno-vla -Werror=write-strings -Wfatal-errors
# in case os linker resolve errors, try adding -mcmodel=large

# Compiler flags supported by GCC but not by the LLVM-based compilers (clang, hipcc, icpx, etc.)
LLVM_UNSUPPORTED_CXXFLAGS := --param vect-max-version-for-alias-checks=50 -Werror=format-contains-nul -Wno-non-template-friend -Werror=return-local-addr -Werror=unused-but-set-variable

CXXFLAGS := -std=c++17 $(HOST_CXXFLAGS) -g
NVCXX_CXXFLAGS := -std=c++20 -O0 -cuda -gpu=managed -stdpar -fpic -gopt
LDFLAGS := -O2 -fPIC -pthread -Wl,-E -lstdc++fs -ldl
LDFLAGS_NVCC := -ccbin $(CXX) --linker-options '-E' --linker-options '-lstdc++fs'
LDFLAGS_NVCXX := -cuda -Wl,-E -ldl -gpu=managed -stdpar

# CUDA
CUDA_BASE := /usr/local/cuda
ifeq ($(wildcard $(CUDA_BASE)),)
# CUDA platform not found
CUDA_BASE :=
else
# CUDA platform at $(CUDA_BASE)
CUDA_LIBDIR := $(CUDA_BASE)/lib64
CUDA_DEPS := $(CUDA_LIBDIR)/libcudart.so
CUDA_ARCH := 60 70 80
CUDA_CXXFLAGS := -I$(CUDA_BASE)/include
CUDA_LDFLAGS := -L$(CUDA_LIBDIR) -lcudart -lcudadevrt
CUDA_NVCC := $(CUDA_BASE)/bin/nvcc
define CUFLAGS_template
$(2)NVCC_FLAGS := $$(foreach ARCH,$(1),-gencode arch=compute_$$(ARCH),code=[sm_$$(ARCH),compute_$$(ARCH)]) -Wno-deprecated-gpu-targets -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --expt-relaxed-constexpr --expt-extended-lambda --generate-line-info --source-in-ptx --display-error-number --threads $$(words $(1)) --cudart=shared
$(2)NVCC_COMMON := -std=c++17 -O3 -g $$($(2)NVCC_FLAGS) -ccbin $(CXX) --compiler-options '$(HOST_CXXFLAGS)'
$(2)CUDA_CUFLAGS := $$($(2)NVCC_COMMON)
endef
$(eval $(call CUFLAGS_template,$(CUDA_ARCH),))
NVCC_COMMON := -std=c++17 -O3 -g $(NVCC_FLAGS) -ccbin $(CXX) --compiler-options '$(HOST_CXXFLAGS)'
CUDA_CUFLAGS := $(NVCC_COMMON)
endif

# ROCm
ROCM_BASE := /opt/rocm
ifeq ($(wildcard $(ROCM_BASE)),)
# ROCm platform not found
ROCM_BASE :=
else
# ROCm platform at $(ROCM_BASE)
ROCM_LIBDIR := $(ROCM_BASE)/lib
ROCM_DEPS := $(ROCM_LIBDIR)/libamdhip64.so
ROCM_ARCH := gfx900 gfx90a gfx1030
ROCM_HIPCC := $(ROCM_BASE)/bin/hipcc
HIPCC_CXXFLAGS := -fno-gpu-rdc $(foreach ARCH,$(ROCM_ARCH),--offload-arch=$(ARCH)) $(filter-out $(LLVM_UNSUPPORTED_CXXFLAGS),$(CXXFLAGS)) --target=$(GCC_TARGET) --gcc-toolchain=$(GCC_TOOLCHAIN) -I$(ROCM_BASE)/include/hip -Wno-unused-result
HIPCC_LDFLAGS := $(LDFLAGS) --target=$(GCC_TARGET) --gcc-toolchain=$(GCC_TOOLCHAIN)
endif

# oneAPI
# Intel GPU ids
ONEAPI_BASE := /opt/intel/oneapi
ifeq ($(wildcard $(ONEAPI_BASE)),)
  # Intel oneAPI not available
  $(warning Cannot find an Intel oneAPI installation at $(ONEAPI_BASE))
  ONEAPI_BASE :=
else
  SYCL_BASE   := $(ONEAPI_BASE)/compiler/latest
  SYCL_LIBDIR := $(SYCL_BASE)/lib
  SYCL_CXX    := $(SYCL_BASE)/bin/icpx
  AOT_CPU_TARGETS   := -fsycl-targets=x86_64
  SYCL_FLAGS        := -fsycl -fp-model=precise
  SYCL_CXXFLAGS := $(filter-out $(LLVM_UNSUPPORTED_CXXFLAGS),$(CXXFLAGS)) $(SYCL_FLAGS) -Wno-unused-variable
  SYCL_LDFLAGS      :=
  # Check for Intel GPU existence
  INTEL_GPU_EXISTS := $(shell sycl-ls | grep -c ext_oneapi_level_zero)
  INTEL_GPU_FLEX := $(shell sycl-ls | grep -c 'GPU Flex')
  # Set AOT_INTEL_TARGETS based on the conditions
  ifneq ($(INTEL_GPU_EXISTS),0)
      ifneq ($(shell sycl-ls | grep -c 'GPU Flex'),0)
          AOT_INTEL_TARGETS := -fsycl-targets=intel_gpu_acm_g10
      else ifneq ($(shell sycl-ls | grep -c 'GPU Max'),0)
          AOT_INTEL_TARGETS := -fsycl-targets=intel_gpu_pvc
      else
          # rely on JIT
          AOT_INTEL_TARGETS :=
      endif
  else
      AOT_INTEL_TARGETS :=
  endif
endif

XTD_BASE := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
CATCH2_BASE := /cvmfs/cms.cern.ch/el8_amd64_gcc12/external/catch2/2.13.6-84f81620c3580a9689570b04155f541c
LIB_INCLUDE := -I$(XTD_BASE)/../include -I$(CATCH2_BASE)/include

SUBDIRS := $(wildcard $(XTD_BASE)/*/)
TARGETS_ALL := $(notdir $(patsubst %/,%,$(SUBDIRS)))

define TEST_template
$(1)/bin:
	mkdir -p $(1)/bin

$(1)Test: $(1)/bin/$(1)_t_cc $(1)/bin/$(1)_t_cuda $(1)/bin/$(1)_t_hip $(1)/bin/$(1)_t_cpusycl $(1)/bin/$(1)_t_gpusycl

$(1)/bin/$(1)_t_cc: $(1)/$(1)_t.cc | $(1)/bin
	$(CXX) $(CXXFLAGS) $(LIB_INCLUDE) $$< -o $$@

$(1)/bin/$(1)_t_cuda: $(1)/$(1)_t.cu | $(1)/bin
	@if [ -z "$(CUDA_BASE)" ]; then \
		echo "Error: CUDA_BASE not set. Skipping $@"; \
	else \
		echo "$(CUDA_NVCC) $(CUDA_CXXFLAGS) $(CUDA_LDFLAGS) $(CUDA_CUFLAGS) $(LIB_INCLUDE) $$< -o $$@"; \
		$(CUDA_NVCC) $(CUDA_CXXFLAGS) $(CUDA_LDFLAGS) $(CUDA_CUFLAGS) $(LIB_INCLUDE) $$< -o $$@; \
	fi

$(1)/bin/$(1)_t_hip: $(1)/$(1)_t.hip.cc | $(1)/bin
	@if [ -z "$(ROCM_BASE)" ]; then \
		echo "Error: ROCM_BASE not set. Skipping $@"; \
	else \
		echo "$(ROCM_HIPCC) $(HIPCC_CXXFLAGS) $(HIPCC_LDFLAGS) $(LIB_INCLUDE) $$< -o $$@"; \
		$(ROCM_HIPCC) $(HIPCC_CXXFLAGS) $(HIPCC_LDFLAGS) $(LIB_INCLUDE) $$< -o $$@; \
	fi

$(1)/bin/$(1)_t_cpusycl: $(1)/$(1)_t.sycl.cc | $(1)/bin
	@if [ -z "$(ONEAPI_BASE)" ]; then \
		echo "Error: ONEAPI_BASE not set. Skipping $@"; \
	else \
		echo "$(SYCL_CXX) $(SYCL_CXXFLAGS) $(SYCL_LDFLAGS) $(AOT_CPU_TARGETS) -DONEAPI_CPU $(LIB_INCLUDE) $$< -o $$@"; \
		$(SYCL_CXX) $(SYCL_CXXFLAGS) $(SYCL_LDFLAGS) $(AOT_CPU_TARGETS) -DONEAPI_CPU $(LIB_INCLUDE) $$< -o $$@; \
	fi

$(1)/bin/$(1)_t_gpusycl: $(1)/$(1)_t.sycl.cc | $(1)/bin
	@if [ -z "$(ONEAPI_BASE)" ]; then \
		echo "Error: ONEAPI_BASE not set. Skipping $@"; \
	else \
		if [ "$(INTEL_GPU_EXISTS)" -ne 0 ]; then \
			echo "$(SYCL_CXX) $(SYCL_CXXFLAGS) $(SYCL_LDFLAGS) $(AOT_INTEL_TARGETS) $(LIB_INCLUDE) $$< -o $$@"; \
			$(SYCL_CXX) $(SYCL_CXXFLAGS) $(SYCL_LDFLAGS) $(AOT_INTEL_TARGETS) $(LIB_INCLUDE) $$< -o $$@; \
		else \
    		echo "Error: Intel GPU not found. Skipping $@"; \
    	fi \
	fi

# List of sin test executables
$(1)_BIN := $(XTD_BASE)/$(1)/bin

# Add targets
TEST_EXECUTABLES := $(1)/bin/$(1)_t_cc
ifdef $(CUDA_BASE)
  TEST_EXECUTABLES += $(1)/bin/$(1)_t_cuda
endif
ifdef $(ROCM_BASE)
  TEST_EXECUTABLES += $(1)/bin/$(1)_t_hip
endif
ifdef $(ONEAPI_BASE)
  TEST_EXECUTABLES += $(1)/bin/$(1)_t_cpusycl
  ifdef $(INTEL_GPU_EXISTS)
    TEST_EXECUTABLES += $(1)/bin/$(1)_t_gpusycl
  endif
endif


#TEST_EXECUTABLES := $$(wildcard $$($(1)_BIN)/*)

# TODO check if sinTest is run or not: is run, should be avoided
run$(1)Test: $(TEST_EXECUTABLES)
	@find $$($(1)_BIN) -maxdepth 1 -type f -exec echo "Running {}" \; -exec {} \; -exec echo \;

clean_$(1)Test:
	rm -rf $(1)/bin
endef
$(foreach target,$(TARGETS_ALL),$(eval $(call TEST_template,$(target))))

clean: $(foreach target,$(TARGETS_ALL), clean_$(target)Test)
all: $(foreach target,$(TARGETS_ALL), $(target)Test)
runAll: $(foreach target,$(TARGETS_ALL), run$(target)Test)
