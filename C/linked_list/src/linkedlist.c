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

struct List* initialize_list(int sorted) {
    struct List* list = (struct List*) malloc(sizeof(struct List));
    list->head = NULL;
    list->sorted = sorted;
    return list;
}

void freeList(struct List* list){
    if (list->head != NULL) {
        struct Node* current = list->head;
        struct Node* next = NULL;
        while (current->next != NULL) {
            next = current->next;
            // printf("freeing %d\n", current->data);
            free(current);
            current = next;
        }
        // printf("freeing %d\n", current->data);
        free(current); 
    }

    free(list);
}

void insert(struct List* list, int data) {
    // create the node
    struct Node* newNode = (struct Node*) malloc(sizeof(struct Node));
    newNode->data = data;
    newNode->next = NULL;

    // add the node into the list
    if (list->head == NULL){
        list->head = newNode;
    
    } else {
        if (list->sorted == 0) {
            struct Node* current = list->head;
            while (current->next != NULL) {
                current = current->next;
            }
            current->next = newNode;
        
        } else if (list->sorted == 1) {
            struct Node* current = list->head;
            while (current->next != NULL) {
                if (current->next->data > data){
                    newNode->next = current->next;
                    current->next = newNode;
                    return;
                }
                current = current->next;
            }
            current->next = newNode;
        }
    }
}

void printList(struct List* list){
    struct Node* current = list->head;
    
    int index = 0; 
    while (current != NULL){
        printf("Node %d = %d\n", index, current->data);
        index++;
        current = current->next;
    }
}

int main() {
    
    struct List* list = initialize_list(0);
    insert(list, 3);
    insert(list, 5);
    insert(list, 4);
    
    printList(list);

    freeList(list);

    return 0;
}