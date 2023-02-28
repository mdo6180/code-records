LEXOR = rdp.l

parse:
	flex $(LEXOR)

# -ll stands for lex libary, some systems it's -lfl
compile: parse
	gcc lex.yy.c -o rdp -ll

test: compile
	./rdp < example-input.txt

clean:
	rm ./rdp lex.yy.c