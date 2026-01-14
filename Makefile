# make <RG_NAME>=<ARG_VALUE>

HDRS := $(wildcard *.h */*.h)
SRCS := $(wildcard *.c */*.c)
BIN  := build/main

run: ${BIN}
	${BIN}

${BIN}: ${SRCS} ${HDRS}
	gcc ${SRCS} -o ${BIN}

clean:
	rm ${BIN}