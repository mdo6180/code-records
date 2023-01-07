# A simple Makefile

flags = -g -std=c++14
gnu_flags = -g

all:
	clang++ $(flags) -o ./bin/main ./src/main.cpp

gnu:
	g++ $(gnu_flags) -o ./bin/main ./src/main.cpp

memory:
	clang++ $(flags) -O2 -ggdb -fsanitize=address -fno-omit-frame-pointer -o ./bin/main ./src/main.cpp

run:
	./bin/main

clean:
	rm -r ./bin/* 