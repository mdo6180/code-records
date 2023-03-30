%{
#include <stdio.h>
#include "types.c"
%}

%token ID CHAR_T INT_T ARRAY_T OF_T 
%token PTR CART MAP
%token NUM LITERAL
%union{
  int value;
  char *name;
  typeptr type;
}
%type <name> ID
%type <type> TYPE E 
%type <value> NUM 
%left MAP
%left CART
%left ','
%left '+'
%right PTR
%left '[' 

%%
P	:	DL ';' E
	;
DL	:	DL ';' D
	|	D
	;
D	:	ID ':' TYPE		{add_symbol($1,$3); }
	;
TYPE	:	TYPE MAP TYPE		{$$ = map($1,$3); }
	|	TYPE CART TYPE		{$$ = cartesian($1,$3); }
	|	ARRAY_T '[' NUM ']' OF_T TYPE  {$$ = array($6,$3); }
	|	PTR TYPE		{$$ = pointer($2); }
	|	'(' TYPE ')'		{$$ = $2; }
	|	CHAR_T			{$$ = new_node(CHAR); }
	|	INT_T			{$$ = new_node(INT); }
	;



E	:	LITERAL		{$$ = new_node(CHAR); }		
	|	NUM		{$$ = new_node(INT);}
	|	ID		{$$ = lookup($1); }		
	|	ID '[' E ']'	{typeptr t;
				t = lookup($1);
				if (t->info == ARRAY) 
				   if ($3->info == INT)
				      $$ = t->p1;
			           else type_error("array index must be integer");
				else type_error("array expected");
				}
	|	ID '(' E ')'	{typeptr t;
				t = lookup($1); 
				if (t->info == MAPPING)
				   if (typematch($3,t->p1))
				     $$ = t->p2;
                                   else type_error("incorrect actuals");
				else type_error("function expected");
				}
	|	E '+' E		{if (($1->info == INT) && ($3->info == INT)) 
					$$ = $1;
				 else type_error("integers expected for +");
				}
	|	E PTR		{ if ($1->info == POINTER) $$ = $1->p1; 
				  else type_error("pointer expected");
				}
	|	E ',' E		{ $$ = cartesian($1,$3); }
	;
%%

#include "lex.yy.c"

yyerror(s) {
    printf("%s line %d\n", s, yylineno);
}

type_error(char *s) {
    printf("type error: %s\n",s); 
    exit(42);
}

typeptr char_node, int_node;
main() {
    yyparse();
}

