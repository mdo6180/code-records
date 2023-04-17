#include <iostream>
#include <memory>


class Node {
public:
    int data;
    std::shared_ptr<Node> left;
    std::shared_ptr<Node> right;
    
    Node(double data, std::shared_ptr<Node> left, std::shared_ptr<Node> right){
        this->data = data;
        this->left = left;
        this->right = right;
    }
    
    ~Node(){
        std::cout << "node " << data << " deleted" << std::endl;
    }
};


int main() {

    std::shared_ptr<Node> node1 = std::make_shared<Node>(1, nullptr, nullptr);

    std::shared_ptr<Node> node2;
    {
        node2 = node1;
        std::cout << "node 1 ref count = " << node1.use_count() << std::endl;
        std::cout << "node 2 ref count = " << node2.use_count() << std::endl;
    }
    
    node2 = nullptr;
    std::cout << "node 1 ref count = " << node1.use_count() << std::endl;
    std::cout << "node 2 ref count = " << node2.use_count() << std::endl;

    {
        std::shared_ptr<Node> node3 = std::make_shared<Node>(3, nullptr, nullptr);
    }

    auto node4 = std::make_shared<Node>(4, nullptr, nullptr);
    auto node6 = std::make_shared<Node>(6, nullptr, nullptr);
    auto node5 = std::make_shared<Node>(5, node4, node6);

    std::cout << "node 4 ref count = " << node4.use_count() << std::endl; 
    std::cout << "node 6 ref count = " << node6.use_count() << std::endl; 
    std::cout << "node 5 ref count = " << node5.use_count() << std::endl; 

    return 0;
}