%{
    #include <string.h>

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

%%

[ \t\n]+    /* remove whitespace */ 

fn {
    output = concat(output, "\n");  
    output = concat(output, yytext); 
    output = concat(output, " "); 
}

[{] {
    output = concat(output, yytext); 
    output = concat(output, "\n"); 
    numtabs++;
    addTabs(numtabs);
}

[}] {
    numtabs--;
    addTabs(numtabs);
    output = concat(output, yytext); 
    output = concat(output, "\n"); 
}

[;] {
    output = concat(output, yytext); 
    output = concat(output, "\n"); 
    newline = 1;
}

== |
[+-=:()<>] |
[a-zA-Z0-9"]+ {
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