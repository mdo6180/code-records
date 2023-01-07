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

    void add(std::vector<int>& data){
        
        for (int i : data){
            auto node = std::make_unique<Node>(i, nullptr);

            if (this->head == nullptr){
                std::cout << "head is null" << std::endl;
                this->head = node.get();
            
            } else {
                std::cout << "current = " << this->head->data << std::endl;
                if (this->sorted == 0){
                    auto current = this->head;
                    while (current->next != nullptr){
                        current = current->next;
                    }
                    current->next = node.get();
                }
                
            }
            std::cout << "head " << this->head->data << std::endl;
        }

        std::cout << "printing list ran" << std::endl;
        auto current = this->head;
        int i = 0;
        while (current->next != nullptr){
            std::cout << "Node " << i << ": " << current->data << std::endl;
            i++;
            current = current->next;
        } 
    }

    void printList(){
        std::cout << "printList() ran" << std::endl;
        auto current = this->head;
        int i = 0;
        while (current->next != nullptr){
            std::cout << "Node " << i << ": " << current->data << std::endl;
            i++;
            current = current->next;
        } 
    }

    ~LinkedList(){
        std::cout << "Linked list deleted" << std::endl;
    }
};

int main(){
    LinkedList list = LinkedList(0);

    std::vector<int> elements = {1,2,3,4,5};
    list.add(elements);
    list.printList();

    return 0;
}
