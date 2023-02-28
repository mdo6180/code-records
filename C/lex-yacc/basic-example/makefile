LEXOR = ch1-02.l

parse:
	lex $(LEXOR)

# -ll stands for "lex library"
# in some systems, replace -ll with -lfl
all: parse
	cc lex.yy.c -o first -ll

run: all
	./first

clean:
	rm ./first lex.yy.c