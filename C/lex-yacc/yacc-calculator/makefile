NAME=calc2

all: $(NAME).tab.c lex.yy.c
	gcc $(NAME).tab.c lex.yy.c -o $(NAME) -ly -ll

$(NAME).tab.c: $(NAME).y
	bison -vd $(NAME).y

lex.yy.c: $(NAME).l
	flex $(NAME).l

clean:
	rm calc2
	rm calc2.output
	rm calc2.tab.c
	rm calc2.tab.h
	rm lex.yy.c