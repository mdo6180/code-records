# A simple Makefile

flags = -g -std=c++14
gnu_flags = -g
BIN_DIR = bin
SRC_DIR = src

all:
	clang++ $(flags) -o $(BIN_DIR)/main $(SRC_DIR)/main.cpp

gnu:
	g++ $(gnu_flags) -o $(BIN_DIR)/main $(SRC_DIR)/main.cpp

run:
	$(BIN_DIR)/main

clean:
	rm -r $(BIN_DIR)/* 