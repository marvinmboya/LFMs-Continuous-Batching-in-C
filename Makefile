# make <RG_NAME>=<ARG_VALUE>

COMPILER := clang
HDRS := $(wildcard *.h */*.h */*/*.h)
SRCS := $(wildcard *.c */*.c */*/*.c)
SRCS := $(filter-out tests/%.c, $(SRCS))
FLAGS := -fopenmp
BIN  := build/main
SHELL_U = $(shell uname -m)

ifeq (${SHELL_U}, arm64)
	ARMPL = /opt/arm/armpl_26.01_flang-21
	ARMFLAGS := -I$(ARMPL)/include -I$(ARMPL)/examples_ilp64
	FLAGS := $(FLAGS) $(ARMFLAGS) 
	FLAGS := $(FLAGS) -mcpu=native -L$(ARMPL)/lib -Wl,-rpath,$(ARMPL)/lib
	LDLIBS := -larmpl_lp64_mp -lamath
else ifeq (${SHELL_U}, x86_64)
	ONEMKL := /opt/intel/oneapi/mkl/2025.0
	MKLLIB := ${ONEMKL}/lib/intel64
	FLAGS := $(FLAGS) -I${ONEMKL}/include
	FLAGS := $(FLAGS) -L${MKLLIB} -Wl,-rpath,${MKLLIB}
	LDLIBS := -lmkl_intel_lp64 -lmkl_sequential -lmkl_core
	LDLIBS := $(LDLIBS) -lpthread -lm -ldl
endif
LDLIBS := $(LDLIBS) -lm

DFLAG ?=
ifeq (${DBG}, True)
	DFLAG := -fsanitize=address -g -O0 
endif 

INCLUDES := -Imodel -Itokenizer

PROMPT ?= What is hello in Spanish
run: ${BIN}
	${BIN} "${PROMPT}"

${BIN}: ${SRCS} ${HDRS}
	${COMPILER} ${DFLAG} ${FLAGS} ${INCLUDES} ${SRCS} -o ${BIN} ${LDLIBS}

clean:
	rm ${BIN}
