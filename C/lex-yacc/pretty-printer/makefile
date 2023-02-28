LEXOR = program1.l

parse:
	flex $(LEXOR)

# -ll stands for lex libary, some systems it's -lfl
compile: parse
	gcc lex.yy.c -o program1 -ll

run0: compile
	./program1 < test0.r > output0-raw.out 
	cat output0-raw.out

run1: compile
	./program1 < test1.r > output1-raw.out
	cat output1-raw.out

run2: compile
	./program1 < test2.r > output2-raw.out
	cat output2-raw.out

run3: compile
	./program1 < test3.r > output3-raw.out
	cat output3-raw.out

clean:
	rm ./program1 lex.yy.c *-raw.out