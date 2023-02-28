#include <iostream>
using namespace std;
#include "tokens.h"

int yylex();

extern int lineno;
int lookahead;

/*
Grammar used:
  Production             Predict
EL --> Ex EL          ( INT
    |                 EOF
Ex --> E =            ( INT
E --> T E'            ( INT
E' --> + T E'         +
    |  - T E'         -
    |                 = )
T --> F T'            ( INT
T' --> * F T'         *
    |  / F T'         /
    |                 = )
F --> ( E )           (
    | INT             INT
*/

void match(int), myerror(char*), EL(), Ex(),E(),T(),F(), E_p(), T_p();

int main() {
  lookahead = yylex();
  EL();
  if (lookahead == 0) /* no more input */
  printf("accepted\n");
  else printf("rejected\n");
}

void match(int token) {
 if (token == lookahead)
    lookahead = yylex();
 else myerror("match");
}

void myerror(char* where) {
  printf("Syntax error line %d: %s\n",lineno, where);
  printf("Token seen: %d\n",lookahead);
  exit(42);
}

void EL() {
  if ((lookahead == LP) || (lookahead == INT)) { Ex(); EL(); }
  else if (lookahead == 0)  /* EOF */
    return;
  else myerror("EL");
}

void Ex() {
  if ((lookahead == LP)||(lookahead==INT)) { E(); match(EQUAL);}
  else myerror("Ex");
}

void E() {
  if ((lookahead == LP)||(lookahead==INT)) { T(); E_p();}
  else myerror("E");
}

void E_p() {
  if (lookahead == PLUS) { match(PLUS); T(); E_p();}
  else if (lookahead == MINUS) { match(MINUS); T(); E_p(); }
  else return;
}

void T() {
  if ((lookahead == LP)||(lookahead==INT)) { F(); T_p();}
  else myerror("E");
}

void T_p() {
  if (lookahead == TIMES ) { match(TIMES); F(); T_p();}
  else if (lookahead == DIVIDE) { match(DIVIDE); F(); T_p(); }
  else return;
}


void F() {
  if (lookahead == LP) {
     match(LP); E(); match(RP); }
  else if (lookahead == INT) match(INT);
  else myerror("F");
}

