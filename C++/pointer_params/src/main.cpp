#include <iostream>
#include <memory>

void changeFirst(std::pair<int,int>* p){
    p->first = 10;
}

void changeSecond(std::pair<int,int>& p){
    p.second = 10;
}

int main(){
    
    std::pair<int,int> pair0 = std::pair<int,int>(130,140);             // standard object
    std::cout << pair0.first << " " << pair0.second << std::endl;
    
    // to pass object to function with * as the parameter
    changeFirst(&pair0);                                                // first get the address of the object, then pass the address to the function
    std::cout << pair0.first << " " << pair0.second << std::endl;
    
    changeSecond(pair0);                                                // pass object to function with & as the parameter
    std::cout << pair0.first << " " << pair0.second << std::endl;
    
    
	
    std::pair<int,int>* pair1 = new std::pair<int,int>(30,40);          // raw pointer
    std::cout << pair1->first << " " << pair1->second << std::endl;
    
    changeFirst(pair1);                                                 // pass raw pointer to function with * as the parameter
    std::cout << pair1->first << " " << pair1->second << std::endl;
    
    // to pass raw pointer to function with & as the parameter
    changeSecond(*pair1);                                               // first dereference the pointer, then pass it by reference to the function
    std::cout << pair1->first << " " << pair1->second << std::endl;
    
    delete pair1;
    
    
    
    auto pair2 = std::make_unique< std::pair<int,int> >(60,80);         // smart pointer
    std::cout << pair2->first << " " << pair2->second << std::endl;
    
    // to pass smart pointer to function with * as the parameter
    changeFirst(pair2.get());                                           // use .get to get the stored pointer, then pass the stored pointer to the function
    std::cout << pair2->first << " " << pair2->second << std::endl;
    
    // to pass smart pointer to function with & as the parameter
    changeSecond(*pair2);                                               // first dereference the pointer, then pass it by reference to the function
    std::cout << pair2->first << " " << pair2->second << std::endl;
    
    std::cout << "done" << std::endl;
	
	return 0;
}
