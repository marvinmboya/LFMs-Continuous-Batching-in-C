# make <RG_NAME>=<ARG_VALUE>

HDRS := $(wildcard *.h */*.h)

SRCS := $(wildcard *.c */*.c */*/*.c)
SRCS := $(filter-out tests/%.c, $(SRCS))
BIN  := build/main

INCLUDES := -Imodel -Itokenizer -Itokenizer/utils

run: ${BIN}
	${BIN}

${BIN}: ${SRCS} ${HDRS}
	gcc ${INCLUDES} ${SRCS} -o ${BIN}

clean:
	rm ${BIN}