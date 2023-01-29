LEXOR = ch1-02.l

parse:
	lex $(LEXOR)

all: parse
	cc lex.yy.c -o first -ll

run: all
	./first

clean:
	rm ./first lex.yy.c