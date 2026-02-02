# make <RG_NAME>=<ARG_VALUE>

COMPILER := clang
HDRS := $(wildcard *.h */*.h */*/*.h)
SRCS := $(wildcard *.c */*.c */*/*.c)
SRCS := $(filter-out tests/%.c, $(SRCS))
FLAGS := -fopenmp
BIN  := build/main
SHELL_U = $(shell uname)

ifeq (${SHELL_U}, Darwin)
	ARMPL = /opt/arm/armpl_26.01_flang-21
	FLAGS := $(FLAGS) -mcpu=native -L$(ARMPL)/lib -Wl,-rpath,$(ARMPL)/lib
	LDLIBS := -larmpl -lamath
endif
LDLIBS := $(LDLIBS) -lm

INCLUDES := -Imodel -Itokenizer

run: ${BIN}
	${BIN}

${BIN}: ${SRCS} ${HDRS}
	${COMPILER} ${FLAGS} ${INCLUDES} ${SRCS} -o ${BIN} ${LDLIBS}

clean:
	rm ${BIN}