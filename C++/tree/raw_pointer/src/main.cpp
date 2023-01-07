#include<iostream>
#include<memory>
#include<queue>

class Node{
public:
    double data = 0.0;
    Node* left = nullptr;
    Node* right = nullptr;
    
    Node(double data, Node* left, Node* right){
        this->data = data;
        this->left = left;
        this->right = right;
    }
    
    ~Node(){
        std::cout << "node " << data << " deleted" << std::endl;
    }
};

class Tree{
public:
    Node* root = nullptr;
    
    Tree(Node* root){
        this->root = root;
    }
    
    ~Tree(){
        deleteTree(root);
        std::cout << "Tree deleted" << std::endl;
    }
    
    void PrintInOrder(){
        PrintInOrder(root);
    }
    
    void insert(double data){
        insert(root, data);
    }
    
    Node* PreOrderSearch(double data){
        return PreOrderSearch(root, data);
    }
    
    Node* BFS(double data){
        return BFS(root, data);
    }
    
    void PrintLevelOrder(){
        PrintLevelOrder(root);
    }
    
private:
    void PrintInOrder(Node* node){
        if(node == nullptr){
            return;
        }
        
        PrintInOrder(node->left);
        std::cout << "data = " << node->data << std::endl;
        PrintInOrder(node->right);
    }
    
    void deleteTree(Node* node){                            // post order deletion is the only viable option
        if(node == nullptr){
            return;
        }
        
        deleteTree(node->left);
        deleteTree(node->right);
        
        delete node;
    }
    
    Node* insert(Node* node, double data){
        if(node == nullptr){
            return new Node(data, nullptr, nullptr);
        }
        
        if(data < node->data){
            node->left = insert(node->left, data);
        }
        
        if(data > node->data){
            node->right = insert(node->right, data);
        }
        
        return node;
    }
    
    Node* PreOrderSearch(Node* node, double data){
        if(node == nullptr){
            return nullptr;
        }
        
        if(node->data == data){
            return node;
        }
        
        if(data < node->data){
            return PreOrderSearch(node->left, data);
        }
        
        if(data > node->data){
            return PreOrderSearch(node->right, data);
        }
        
        return node;
        
        /*
         if(node == nullptr){
            return nullptr;
         }
         
         if(node->data == data){
            return node;
         }
         
         Node* left = PreOrderSearch(node->left, data);
         Node* right = PreOrderSearch(node->right, data);
         
         if(left != nullptr){
            return left;
         
         } else if(right != nullptr){
            return right;
         
         } else {
            return nullptr;
         }
         
         return node;
         */
    }
    
    Node* BFS(Node* node, double data){
        
        if(node == nullptr){
            return nullptr;
        }
        
        std::queue<Node*> Queue;
        
        Queue.push(node);
        
        while ( !Queue.empty() ) {
            Node* n = Queue.front();
            
            if(n->data == data){
                return n;
            }
            
            if(n->left != nullptr){
                Queue.push(n->left);
            }
            
            if(n->right != nullptr){
                Queue.push(n->right);
            }
            
            Queue.pop();
        }
        
        return nullptr;
    }
    
    void PrintLevelOrder(Node* node){
        
        if(node == nullptr){
            std::cout << "Root is null" << std::endl;
            return;
        }
        
        std::queue<Node*> Queue;
        
        Queue.push(node);
        
        while ( !Queue.empty() ) {
            Node* n = Queue.front();
            
            std::cout << n->data << std::endl;
            
            if(n->left != nullptr){
                Queue.push(n->left);
            }
            
            if(n->right != nullptr){
                Queue.push(n->right);
            }
            
            Queue.pop();
        }
    }
    
};


int main(){
    
    Node* node15 = new Node(1.5, nullptr, nullptr);
    Node* node52 = new Node(5.2, nullptr, nullptr);
    Node* node77 = new Node(7.7, nullptr, nullptr);
    Node* node62 = new Node(6.2, node52, node77);
    Node* node45 = new Node(4.5, nullptr, node62);
    Node* node35 = new Node(3.5, node15, node45);
    
    Tree bst = Tree(node35);
    
    bst.PrintLevelOrder();
    
    std::cout << "---------------------------" << std::endl;
    
    bst.insert(8.8);

    for(double i = 0.3; i < 7.0; i+=1.5){
        bst.insert(i);
    }
    
    bst.PrintInOrder();            // print function causes problems because unique pointer in  insert method is detroyed prematurely
    
    std::cout << "hey there" << std::endl;
    
    return 0;
}
