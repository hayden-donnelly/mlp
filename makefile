.PHONY: all clean run

all: build/cpu_mlp

build/cpu_mlp: src/main.c
	mkdir -p build
	gcc -o cpu_mlp src/main.c -lm
	mv cpu_mlp build/cpu_mlp

run: build/cpu_mlp
	./build/cpu_mlp

clean:
	rm -rf build
