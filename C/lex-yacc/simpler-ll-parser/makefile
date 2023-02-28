LEXOR = rdp.l

parse:
	flex $(LEXOR)

# -ll stands for lex libary, some systems it's -lfl
compile: parse
	gcc lex.yy.c -o rdp -ll

test-accept: compile
	./rdp < example-accept.txt

test-reject: compile
	./rdp < example-reject.txt

clean:
	rm ./rdp lex.yy.c