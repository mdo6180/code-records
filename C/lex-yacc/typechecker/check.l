%%
char		{return(CHAR_T);}
integer		{return(INT_T);}
array		{return(ARRAY_T);}
of		{return(OF_T);}
\,		{return(',');}
\;		{return(';');}
\[		{return('[');}
\]		{return(']');}
\(		{return('(');}
\)		{return(')');}
\:		{return(':');}
\^		{return(PTR);}
\+		{return('+');}
X		{return(CART);}
\-\>		{return(MAP);}
\'[a-z]\'	{return(LITERAL);}
[a-z]+		{yylval.name = (char*)malloc(strlen(yytext)+1);
                 strcpy(yylval.name,yytext); return(ID);}
[0-9]*		{yyval.value = atoi(yytext);return(NUM);}
[ \t\n]		{}
%%