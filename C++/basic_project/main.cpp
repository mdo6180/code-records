#include <iostream>
#include <vector>
#include <string>


int main() {
    std::vector<std::string> msg {"hello", "there", "friend"};

    for (const std::string& word : msg) {
        std::cout << word << std::endl;
    }

    std::cout << "Hello World!" << std::endl;
    return 0;
}
