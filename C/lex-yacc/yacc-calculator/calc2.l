%{
#include "calc2.tab.h"
%}

%%
"+"	    {return(PLUS);}
"*"	    {return(TIMES);}
"("	    {return(LPAREN);}
")"	    {return(RPAREN);}
[\n]	{return(CR);}
[0-9]+	{yylval=atoi(yytext); return(INT);}
[\t ]	;
%%