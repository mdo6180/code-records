BIN=bin

# makefile syntax
# target: prerequisites
# <TAB> recipe

# this target to create bin directory so we can store our binaries and exclude from version control
$(BIN):
	if [ ! -d "./$(BIN)" ]; then \
		mkdir ./$(BIN); \
	fi 

# compile target requires $(BIN) target to run
compile: $(BIN)
	gcc -o ./$(BIN)/linkedlist ./src/linkedlist.c

# run target runs the executable, requires compile target to execute first (i.e., automatically compiles first)
run: compile
	./bin/linkedlist

debug:
	gcc -g -o ./$(BIN)/linkedlist ./src/linkedlist.c

clean:
	rm $(BIN)/linkedlist