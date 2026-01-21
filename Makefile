# make <RG_NAME>=<ARG_VALUE>

COMPILER := clang
HDRS := $(wildcard *.h */*.h */*/*.h)
SRCS := $(wildcard *.c */*.c */*/*.c)
SRCS := $(filter-out tests/%.c, $(SRCS))
FLAGS := -fopenmp
BIN  := build/main

INCLUDES := -Imodel -Itokenizer

run: ${BIN}
	${BIN}

${BIN}: ${SRCS} ${HDRS}
	${COMPILER} ${FLAGS} ${INCLUDES} ${SRCS} -o ${BIN}

clean:
	rm ${BIN}