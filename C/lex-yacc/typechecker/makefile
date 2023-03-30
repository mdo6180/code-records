typecheck: lex.yy.c y.tab.c types.h types.c 
	gcc -o typecheck y.tab.c -ly -ll

lex.yy.c: check.l
	lex check.l

y.tab.c: check.y 
	yacc check.y

clean:
	rm y.tab.c
	rm lex.yy.c