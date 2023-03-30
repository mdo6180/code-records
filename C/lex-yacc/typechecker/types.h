typedef enum {CHAR,INT,ARRAY,POINTER,CARTESIAN,MAPPING,UNKNOWN} type_info;
typedef struct tnode {
    type_info info;
    struct tnode *p1, *p2;
    int array_size;
} typenode, *typeptr;

typeptr new_node(type_info info);
typeptr map(),cartesian();
typeptr lookup();
int typematch();

typedef struct s {
    char *name;
    typeptr t;
    struct s *next;
} typetable;