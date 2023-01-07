#include<iostream>
#include<memory>

class Node{
public:
    double data;
    Node* left;
    Node* right;
    
    Node(double data, Node* left, Node* right){
        this->data = data;
        this->left = left;
        this->right = right;
        //std::cout << "node " << data << " created" << std::endl;
    }
    
    ~Node(){
        std::cout << "node " << data << " deleted" << std::endl;
    }
};

class Tree{
public:
    Node* root;
    
    Tree(Node* root){
        this->root = root;
    }
    
    ~Tree(){
        //deleteTree(root);
        std::cout << "Tree deleted" << std::endl;
    }
    
    void PrintInOrder(){
        PrintInOrder(root);
    }
    
    void insert(Node* newNode){
        insert(root, newNode);
    }
    
    Node* PreOrderSearch(double data){
        return PreOrderSearch(root, data);
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
    
    Node* insert(Node* node, Node* newNode){
        if(node == nullptr){
            return newNode;
            //auto newNode = std::make_unique<Node>(data, nullptr, nullptr);
            //return newNode.get();
        }
        
        if(newNode->data < node->data){
            node->left = insert(node->left, newNode);
        }
        
        if(newNode->data > node->data){
            node->right = insert(node->right, newNode);
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
};


int main(){
    
    auto node15 = std::make_unique<Node>(1.5, nullptr, nullptr);
    auto node52 = std::make_unique<Node>(5.2, nullptr, nullptr);
    auto node77 = std::make_unique<Node>(7.7, nullptr, nullptr);
    auto node62 = std::make_unique<Node>(6.2, node52.get(), node77.get());
    auto node45 = std::make_unique<Node>(4.5, nullptr, node62.get());
    auto root35 = std::make_unique<Node>(3.5, node15.get(), node45.get());
    
    auto insertNode = std::make_unique<Node>(8.8, nullptr, nullptr);
    
    Tree bst = Tree(root35.get());
    
    bst.insert(insertNode.get());

    /*
    std::unique_ptr<Node> empty_ptr;
    for(double i = 0.3; i < 7.0; i+=1.5){
    	empty_ptr = std::make_unique<Node>(i, nullptr, nullptr);
    	bst.insert(empty_ptr.get());
    	//i += 1.5;
    }
    */
    
    std::cout << "hello" << std::endl;
    
    bst.PrintInOrder();            // print function causes problems because unique pointer in  insert method is detroyed prematurely
    
    return 0;
}
