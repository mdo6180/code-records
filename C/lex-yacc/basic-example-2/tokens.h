#define PLUS 1
#define MINUS 2
#define TIMES 3
#define DIV 4
#define MOD 5
#define ASSIGN 6
#define LP 7
#define RP 8
#define LB 9
#define RB 10
#define COLON 11
#define COMMA 12
#define SEMI 13
#define RET_SYM 14
#define I32_T 15
#define FN_T 16
#define PRINTLN_T 17
#define IF_T 18
#define ELSE_T 19
#define THEN_T 20
#define LET_T 21
#define MUT_T 22
#define INT_LIT 23
#define ID 24

void match(int);
void myerror(char*);
void rusty();
void functions();
void functions_p();
void function();
void opt_parameter();
void params();
void params_p();
void param();
void opt_return();
void statements();
void statements_p();
void statement();
void LET_T_p();
void ELSE_T_p();
void expr();
void expr_p();
void term();
void term_p();
void factor();
void factor_p();
void opt_actuals();
void actuals();
void actuals_p();