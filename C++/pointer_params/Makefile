# A simple Makefile

flags = -g -std=c++17

all:
	clang++ $(flags) -o ./bin/main ./src/main.cpp

memory:
	clang++ -std=c++17 -O1 -fsanitize=address -fno-omit-frame-pointer -o ./bin/main ./src/main.cpp

clean:
	rm -r ./bin/* 