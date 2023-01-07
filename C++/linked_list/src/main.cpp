#include<iostream>
#include<memory>
#include <vector>

class Node{
public:
    int data;
    Node* next;

    // contructor 
    Node(int data, Node* next){
        this->data = data;
        std::cout << "node " << data << " created" << std::endl;
    }
    
    // destructor
    ~Node(){
        std::cout << "node " << data << " deleted" << std::endl;
    }
};

class LinkedList{
public:
    Node* head;
    int sorted;

    LinkedList(int sorted){
        this->head = nullptr;
        this->sorted = sorted;
    }

    void add(int data){
        Node* newNode = new Node(data, nullptr);

        if (this->head == nullptr){
            this->head = newNode;
        
        } else {
            if (this->sorted == 0) {
                Node* current = this->head;
                while (current->next != NULL) {
                    current = current->next;
                }
                current->next = newNode;

            } else if (this->sorted == 1) {
                Node* current = this->head;
                while (current->next != nullptr) {
                    if (current->next->data > data) {
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

    void printList(){
        Node* current = this->head;
    
        int index = 0; 
        while (current != nullptr) {
            std::cout << "Node " << index << " = " << current->data << std::endl;
            index++;
            current = current->next;
        }
    }

    ~LinkedList(){
        if (this->head != nullptr) {
            Node* current = this->head;
            Node* next = nullptr;

            while (current->next != nullptr) {
                next = current->next;
                delete current;
                current = next;
            }
            
            delete current;
        }
        
        std::cout << "Linked list deleted" << std::endl;
    }
};

int main(){
    LinkedList list = LinkedList(1);
    list.add(1);
    list.add(2);
    list.add(6);
    list.add(5);
    list.add(4);

    list.printList();

    return 0;
}
