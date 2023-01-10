mod linked_list;
use linked_list::singly::{SLL, SLLNode};

fn main() {
    let mut list: SLL = SLL::new();
    let node3: SLLNode = SLLNode::new(3, None);
    let node2: SLLNode = SLLNode::new(2, Some(Box::new(node3)));
    let node1: SLLNode = SLLNode::new(1, Some(Box::new(node2)));
    list.head = Some(Box::new(node1));

    list.insert(5);
    list.insert(6);
    list.print_list();
}

/*
    fn check_optional(optional: Option<Box<SLLNode>>) {
        match optional {
            Some(p) => println!("has value {}", p.value),
            None => println!("has no value"),
        }
    }

    check_optional(list.head);

struct Rectangle {
    width: i32,
    height: i32,
}

impl Rectangle {
    fn area(rectangle: &Rectangle) -> i32 {
        rectangle.width * rectangle.height
    }
}

struct Node {
    value: i32,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>
}



fn main() {
    let rect1 = Rectangle{
        width: 30, 
        height: 40
    };

    println!(
        "The area of the rectangle is {} square pixels.",
        Rectangle::area(&rect1)
    );

    let node2 = Node{
        value: 2,
        left: None,
        right: None 
    };

    let node3 = Node{
        value: 3,
        left: None,
        right: None 
    };

    let node1 = Node{
        value: 1,
        left: Some(Box::new(node2)),
        right: Some(Box::new(node3))
    };

    fn check_optional(optional: Option<Box<Node>>) {
        match optional {
            Some(p) => println!("has value {}", p.value),
            None => println!("has no value"),
        }
    }

    check_optional(node1.left);
}
*/
