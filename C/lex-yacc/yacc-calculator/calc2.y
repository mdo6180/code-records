%{

/*  
All of the below productions that do not have associated 
actions are using the DEFAULT action -- $$ = $1 
*/

%}
%token PLUS TIMES INT CR RPAREN LPAREN

%%
lines	:	lines line
        |	line
        ;
line	:	expr CR 		    {printf("= %d\n",$1); }
        ;
expr	:	expr PLUS term 		{$$ = $1 + $3; }
        |	term
        ;
term	:	term TIMES factor	{$$ = $1 * $3; }
        |	factor
        ;
factor	:	LPAREN expr RPAREN	{$$ = $2;}
        |	INT
        ;
%%