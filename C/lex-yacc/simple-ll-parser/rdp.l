%{
    #define PLUS 1
    #define MINUS 2
    #define TIMES 3
    #define DIVIDE 4
    #define LP 5
    #define RP 6
    #define INT 7
    #define EQUAL 8

    int lineno = 1;
    int lookahead;
%}

%%
[ \t]+    /* remove whitespace */ 
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

void match(int), myerror(char*), EL(), Ex(),E(),T(),F(), E_p(), T_p();

int main() {
    lookahead = yylex();
    EL();
    if (lookahead == 0) {       /* no more input */
        printf("accepted\n");
    } else {
        printf("rejected\n");
    }
    printf("EOF = %d\n", EOF);
}

void match(int token) {
    if (token == lookahead) {
        lookahead = yylex();
    } else {
        myerror("match");
    }
}

void myerror(char* where) {
    printf("Syntax error line %d: %s\n", lineno, where);
    printf("Token seen: %d\n",lookahead);
    exit(42);
}

void EL() {
    if ((lookahead == LP) || (lookahead == INT)) { 
        Ex(); 
        EL(); 
    } else if (lookahead == 0) {        /* EOF */
        return;
    } else {
        myerror("EL");
    }
}

void Ex() {
    if ((lookahead == LP)||(lookahead == INT)) { 
        E(); 
        match(EQUAL);
    } else {
        myerror("Ex");
    }
}

void E() {
    if ((lookahead == LP)||(lookahead == INT)) { 
        T(); 
        E_p();
    } else {
        myerror("E");
    }
}

void E_p() {
    if (lookahead == PLUS) { 
        match(PLUS); 
        T(); 
        E_p();
    } else if (lookahead == MINUS) { 
        match(MINUS); 
        T(); 
        E_p(); 
    } else {
        return;
    }
}

void T() {
    if ((lookahead == LP)||(lookahead == INT)) { 
        F(); 
        T_p();
    }
    else {
        myerror("E");
    }
}

void T_p() {
    if (lookahead == TIMES ) { 
        match(TIMES); 
        F(); 
        T_p();
    }
    else if (lookahead == DIVIDE) { 
        match(DIVIDE); 
        F(); 
        T_p(); 
    } else {
        return;
    }
}

void F() {
    if (lookahead == LP) {
        match(LP); 
        E(); 
        match(RP); 
    } else if (lookahead == INT) {
        match(INT);
    } else {
        myerror("F");
    }
}