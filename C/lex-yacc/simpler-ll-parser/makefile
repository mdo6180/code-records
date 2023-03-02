LEXOR = rdp.l

parse:
	flex $(LEXOR)

# -ll stands for lex libary, some systems it's -lfl
compile: parse
	gcc lex.yy.c -o rdp -ll

test-accept-1: compile
	./rdp < example-accept-1.txt

test-accept-2: compile
	./rdp < example-accept-2.txt

test-reject: compile
	./rdp < example-reject.txt

clean:
	rm ./rdp lex.yy.c