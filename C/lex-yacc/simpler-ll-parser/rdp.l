%{
    int lineno = 1;
    int lookahead;
%}

%%
[ \t]+    /* remove whitespace */ 
[a]	{return('a');}
[b] {return('b');}
[c] {return('c');}
[\n] 	lineno++;
.	;
%%

void match(char), myerror(char*), S(), B(),C();

int main() {
    lookahead = yylex();
    S();
    if (lookahead == 0) {       /* no more input */
        printf("accepted\n");
    } else {
        printf("rejected\n");
    }
}

void match(char symbol) {
    if (symbol == lookahead) {
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

void S() {
    if (lookahead == 'a') { 
        match('a');
        B(); 
    } else if (lookahead == 'b') { 
        match('b'); 
        C(); 
    } else {
        myerror("expecting a or b");
    }
}

void B() {
    if (lookahead == 'b') {
        match('b'); 
        match('b'); 
        C();
    } else {
        myerror("match B");
    }
}

void C() {
    if (lookahead == 'c') {
        match('c');
        match('c');
    } else {
        myerror("match C"); 
    }
}