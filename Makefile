# make <RG_NAME>=<ARG_VALUE>

HDRS := $(wildcard *.h */*.h)
SRCS := $(wildcard *.c */*.c)
BIN  := build/main

INCLUDES := -Imodel -Itokenizer

run: ${BIN}
	${BIN}

${BIN}: ${SRCS} ${HDRS}
	gcc ${INCLUDES} ${SRCS} -o ${BIN}

clean:
	rm ${BIN}