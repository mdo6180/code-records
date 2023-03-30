#include "types.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>


typeptr new_node(type_info info) {
  typeptr t;
  t = (typeptr)malloc(sizeof(typenode));
  t->info = info;
  t->p1 = NULL;
  t->p2 = NULL;
  return t;
}

int typematch(typeptr t1, typeptr t2) {
    int v;
    if (t1 && t2) 
    if (t1->info == t2->info) 
        switch (t1->info) {
            case CHAR: return 1;
            case INT: return 1;
            case ARRAY: if (t1->array_size == t2->array_size) 
                            return typematch(t1->p1,t2->p1);
                else return 0;
            case POINTER: return typematch(t1->p1, t2->p1); 
            case CARTESIAN: v = typematch(t1->p1, t2->p1);
                            if (v) return typematch(t1->p2,t2->p2);
                            else return 0;
            case MAPPING: v = typematch(t1->p1, t2->p1);
                            if (v) return typematch(t1->p2,t2->p2);
                            else return 0;
            case UNKNOWN: return 0;
        }  /* switch */
        else return 0;
    else if (!t1 && !t2) return 1;
    else return 0;
}

typeptr map(typeptr t1,typeptr t2) {
    typeptr t;
    t = (typeptr)malloc(sizeof(typenode));
    t->info = MAPPING;
    t->p1 = t1;
    t->p2 = t2;
    return t;
}

typeptr cartesian(typeptr t1,typeptr t2) {
    typeptr t;
    t = (typeptr)malloc(sizeof(typenode));
    t->info = CARTESIAN;
    t->p1 = t1;
    t->p2 = t2;
    return t;
}

typeptr array(typeptr t1,int size) {
    typeptr t;
    t = (typeptr)malloc(sizeof(typenode));
    t->info = ARRAY;
    t->p1 = t1;
    t->p2 = NULL;
    t->array_size = size;
    return t;
}

typeptr pointer(typeptr t1) {
    typeptr t;
    t = (typeptr)malloc(sizeof(typenode));
    t->info = POINTER;
    t->p2 = NULL;
    t->p1 = t1;
    return t;
}

typetable *type_table = NULL;

void add_symbol(char *name,typeptr type) {
    typetable *t;
    t = (typetable*)malloc(sizeof(typetable));
    t->name = (char*)malloc(strlen(name)+1);
    strcpy(t->name,name);
    t->t = type;
    t->next = type_table;
    type_table = t; 
}

typeptr lookup(char *name) {
    typetable *t;
    t = type_table;
    while (t) {
        if (strcmp(t->name,name) ==0) return t->t;
        t = t->next;
    }
    return NULL;
}