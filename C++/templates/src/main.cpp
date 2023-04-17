#include <iostream>
#include <memory>
#include <cmath>


template<typename T> class Node {
public:
    T data;
    std::shared_ptr<Node<T>> next;
    
    // By defining the constructor to take a const T& parameter instead of a T parameter,
    // we allow the constructor to be called with a pre-constructed T object, and avoids the need for a default constructor for the T class.
    Node(const T& data, const std::shared_ptr<Node<T>>& next) : data(data), next(next) {}
    
    ~Node() {
        std::cout << data << " object destroyed" << std::endl;
    }
};

class Shape {
public:
    // "Pure virtual function" is the C++ technically correct term 
    // which specifically denotes the fact that the function is set to 0.
    virtual const float area() const = 0;

    bool operator< (const Shape& other) {
        if (this->area() < other.area()) {
            return true;
        } else {
            return false;
        }
    }

    bool operator> (const Shape& other) {
        if (this->area() > other.area()) {
            return true;
        } else {
            return false;
        }
    }

    bool operator== (const Shape& other) {
        if (this->area() == other.area()) {
            return true;
        } else {
            return false;
        }
    }
};

class Circle : public Shape {
public:
    float radius;

    Circle(float radius) {
        this->radius = radius;
    }

    const float area() const {
        return M_PI * pow(this->radius, 2);
    }
    
    // The << operator is overloaded as a friend function, 
    // which allows it to access private members of the MyClass object.
    friend std::ostream& operator<< (std::ostream& os, const Circle& circle) {
        std::cout << "Circle(radius=" << circle.radius << ", area=" << circle.area() << ")";
        return os;
    }
};

class Rectangle : public Shape {
public:
    float length;
    float width;

    Rectangle(float length, float width) {
        this->length = length;
        this->width = width;
    }

    const float area() const {
        return this->length * this->width;
    }

    friend std::ostream& operator<< (std::ostream& os, const Rectangle& rect) {
        std::cout << "Rectangle(length=" << rect.length << ", width=" << rect.width << ", area=" << rect.area() << ")";
        return os;
    }   
};

int main() {
    Circle circle1(5.0);
    Circle circle2(6.0);
    Circle circle3(5.0);

    if (circle1 < circle2) {
        std::cout << "circle1 < circle2" << std::endl;
    } 
    
    if (circle2 > circle1) {
        std::cout << "circle2 > circle1" << std::endl; 
    }

    if (circle1 == circle3) {
        std::cout << "circle1 == circle3" << std::endl;  
    }

    std::cout << circle1 << std::endl;

    Rectangle rect1(3.0, 5.0);
    Rectangle rect2(5.0, 5.0);

    if (rect1 < rect2) {
        std::cout << "rect1 < rect2" << std::endl;    
    } 
    std::cout << rect1 << std::endl;

    if (circle1 > rect1) {
        std::cout << circle1 << " is larger than " << rect1 << std::endl;
    }

    Node<Circle> node_circle = Node<Circle>(circle1, nullptr);

    return 0;
}
