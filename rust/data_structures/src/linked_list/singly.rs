use std::option::Option;

pub struct SLLNode {
    pub value: i32,
    pub next: Option<Box<SLLNode>>,
}

pub struct SinglyLinkedList {
    pub head: Option<Box<SLLNode>>,
    pub num_elements: i32
}

impl SLLNode {
    pub fn new(value: i32, next: Option<Box<SLLNode>>) -> SLLNode {
        SLLNode {
            value: value,
            next: next
        }
    }
}

impl SinglyLinkedList {
    pub fn new() -> SinglyLinkedList {
        SinglyLinkedList {
            head: None,
            num_elements: 0
        }
    }

    pub fn insert(&mut self, value: i32) {
        // is_none() method is part of std::option::Option
        if self.head.is_none() {
            self.head = Some(
                Box::new(
                    SLLNode::new(value, None)
                )
            );
            self.num_elements += 1;
        
        } else {
            // use .as_mut() to create a &mut (mutable reference) to the Box value (aka the head)
            let mut current = self.head.as_mut();

            loop {
                match current {
                    Some(node) => {
                        // checks if the next node is None,
                        // if it is not None, set current to a &mut of the next node.
                        // if it is None, we are at the end of the list, 
                        // thus, set the next parameter to another node
                        if node.next.is_some() {
                            current = node.next.as_mut();
                        } else {
                            node.next = Some(Box::new(SLLNode::new(value, None)));
                            self.num_elements += 1;
                            break;
                        }
                    },
                    None => {
                        break;
                    }
                };
            }
        }
    }

    pub fn print_list(&self) {
        if self.head.is_some() {
            let mut current = &self.head;
            let mut position = 0;

            loop {
                match current {
                    Some(node) => {
                        println!("element {} = {}", position, node.value);
                        current = &node.next;
                        position += 1;
                    },
                    None => {
                        break;
                    }
                };
            }
        } else {
            println!("List is empty");
        }
    }
}

/*
fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}
*/