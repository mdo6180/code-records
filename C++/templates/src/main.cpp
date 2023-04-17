#include <iostream>
#include <memory>
#include <cmath>


template<typename T> class Node {
public:
    T data;
    std::shared_ptr<Node> next;
    ~Node();
};

class Shape {
public:
    const float area() const {};

    bool operator< (const Shape& other) {
        if (this->area() < other.area()) {
            return true;
        } else {
            return false;
        }
    }
};

class Circle {
public:
    float radius;

    Circle(float radius) {
        this->radius = radius;
    }

    bool operator< (const Circle& other) {
        if (radius < other.radius) {
            return true;
        } else {
            return false;
        }
    }
    
    bool operator> (const Circle& other) {
        if (radius > other.radius) {
            return true;
        } else {
            return false;
        }
    }

    bool operator== (const Circle& other) {
        if (radius == other.radius) {
            return true;
        } else {
            return false;
        }
    }
    
    // The << operator is overloaded as a friend function, 
    // which allows it to access private members of the MyClass object.
    friend std::ostream& operator<< (std::ostream& os, const Circle& other) {
        std::cout << "Circle(radius=" << other.radius << ")";
        return os;
    }
};

class Rectangle {
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

    bool operator< (const Rectangle& other) {
        if (this->area() < other.area()) {
            return true;
        } else {
            return false;
        }
    } 

    friend std::ostream& operator<< (std::ostream& os, const Rectangle& rect) {
        std::cout << "Rectangle(length=" << rect.length << ", width=" << rect.width << ")";
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

    Rectangle rect1(3.0, 5.0);
    Rectangle rect2(5.0, 5.0);

    if (rect1 < rect2) {
        std::cout << "rect1 < rect2" << std::endl;    
    } 
    std::cout << rect1 << std::endl;

    return 0;
}
