LEXOR = program2.l
BIN = program2

lex.yy.c:
	flex $(LEXOR)

# -ll stands for lex libary, some systems it's -lfl
compile: lex.yy.c tokens.h
	gcc -o program2 lex.yy.c -ll

run0: compile
	./program2 < test0.r > output0-raw.out 
	cat output0-raw.out

run1: compile
	./program2 < test1.r > output1-raw.out
	cat output1-raw.out

run2: compile
	./program2 < test2.r > output2-raw.out
	cat output2-raw.out

run3: compile
	./program2 < test3.r > output3-raw.out
	cat output3-raw.out

clean:
	rm ./program2 lex.yy.c *-raw.out