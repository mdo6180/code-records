#include <stdio.h>
#include <stdlib.h>

struct Node {
    int data;
    struct Node* next;
};

struct List{
    struct Node* head;
    int sorted;
};

void insert(struct List* list, int data) {

    struct Node* current = list->head;

    if (current == NULL) {
        list->head = (struct Node*) malloc(sizeof(struct Node));
        list->head->data = data;
        list->head->next = NULL;
    
    } else {
            
    }

    /*
    if (current != NULL) {
        printf(" %d ", current->data);
    }
    */

    // if list is empty, insert the first element as the head of the list
    /*
    if (list->head == NULL) {
        list->head = (struct Node*) malloc(sizeof(struct Node));
        list->head->data = data;
        list->head->next = NULL;
    }
    */

    /*
    while (head != NULL) {
        
    }
    
    if (list->sorted == 0) {
        printf("unsorted\n");

    } else if (list->sorted == 1) {
        printf("sorted\n");

    } else {
        printf("must be a boolean");
    }
    
    */
    // printf(" %d ", list->head->data);
}

void freeList(struct List* list) {
    free(list->head);
    free(list);
    printf("done.\n");
}

int main() {
    
    struct List* list = (struct List*) malloc(sizeof(struct List));
    list->head = (struct Node*) malloc(sizeof(struct Node));
    /*
    list->head->data = 3;
    list->head->next = NULL;
    */
    list->head = NULL;
    list->sorted = 1;
    insert(list, 3);

    freeList(list);

    return 0;
}
