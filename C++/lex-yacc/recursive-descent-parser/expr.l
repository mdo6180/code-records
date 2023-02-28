%{
#include <iostream>
using namespace std;
#include "tokens.h"
int lineno=1;
%}
%%
\+	{return(PLUS);}
\-	{return(MINUS);}
\*	{return(TIMES);}
\/	{return(DIVIDE);}
\(	{return(LP);}
\)	{return(RP);}
\=	{return(EQUAL);}
[0-9]*	{return(INT);}
[\n] 	lineno++;
.	;
%%