compute: lex.yy.c rdp_parser.cpp tokens.h
	g++ -x c++ -o compute rdp_parser.cpp lex.yy.c -ll

lex.yy.c: expr.l
	flex expr.l

clean:
	rm lex.yy.c compute