%{
    #include <string.h>
    #include "tokens.h"

    #define PLUS 1
    #define MINUS 2
    #define TIMES 3
    #define DIVIDE 4
    #define LP 5
    #define RP 6
    #define INT 7
    #define EQUAL 8

    char* output = "\n";
    int numtabs = 0;
    int newline = 0;
    int double_quotes = 0;

    char* concat(char* str1, char* str2) {
        /* calculate the length of the new string */
        size_t len = strlen(str1) + strlen(str2);

        /* allocate memory for the new string */
        char* str = malloc(len + 1);

        /* concatenate */
        strcpy(str, str1);
        strcat(str, str2);

        return str;
    }

    void addTabs(int numtabs) {
        int i;
        for (i = 0; i < numtabs; i++) {
            output = concat(output, "   ");
        }
    }
%}

FN_T fn
ID [a-zA-Z0-9"]+
I32_T i32
LET_T let
IF_T if
ELSE_T else
MUT_T mut
PRINTLN_T println
THEN_T then
INT_LIT [0-9]
COMMA ,
PLUS [+]
MINUS -
MOD %
ASSIGN =
LP [(]
RP [)]
LB [{]
RB [}]
SEMI ;
COLON :
TIMES [*]
DIV [/]
COMMENT \/{2}
RET_SYM ->
COMPARE ==
LESS <
MORE >

%%

[ \t\n]+    /* remove whitespace */ 

{FN_T} {
    output = concat(output, "\n");  
    output = concat(output, yytext); 
    output = concat(output, " "); 
}

{LB} {
    output = concat(output, yytext); 
    output = concat(output, "\n"); 
    numtabs++;
    addTabs(numtabs);
}

{RB} {
    numtabs--;
    addTabs(numtabs);
    output = concat(output, yytext); 
    output = concat(output, "\n"); 
}

{SEMI} {
    output = concat(output, yytext); 
    output = concat(output, "\n"); 
    newline = 1;
}

{COMMENT} |
{COMPARE} |
{ASSIGN} |
{PLUS} |
{MINUS} |
{TIMES} |
{DIV} |
{MOD} |
{COLON} |
{LP} |
{RP} |
{LESS} |
{MORE} |
{ID} {
    if (newline == 1) {
        addTabs(numtabs);
        newline = 0;
    }
    output = concat(output, yytext); 
    output = concat(output, " "); 
}

%%

int main() { 
    yylex();
    printf("%s\n", output);
}