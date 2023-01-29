use std::option::Option;
use std::fmt;

pub struct SLLNode {
    pub value: Option<i32>,
    pub next: Option<Box<SLLNode>>,
}

pub struct SinglyLinkedList {
    pub head: Option<Box<SLLNode>>,
    pub num_elements: i32,
    pub sorted: bool
}

impl SLLNode {
    pub fn new(value: Option<i32>, next: Option<Box<SLLNode>>) -> SLLNode {
        SLLNode {
            value: value,
            next: next
        }
    }
}

impl fmt::Display for SLLNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.value {
            Some(val) => {
                write!(f, "Node({})", val)
            },
            None => {
                write!(f, "Node(None)")
            }
        }
    }
}

impl SinglyLinkedList {
    pub fn new(head: Option<Box<SLLNode>>, sorted: bool) -> SinglyLinkedList {
        SinglyLinkedList {
            head: head,
            num_elements: 0,
            sorted: sorted
        } 
    }

    pub fn insert(&mut self, value: i32) {
        if self.head.is_none() {
            self.head = Some(
                Box::new(
                    SLLNode::new(Some(value), None)
                )
            );
            self.num_elements += 1;
        
        } else {
            {
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
                                node.next = Some(Box::new(SLLNode::new(None, None)));
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
            
            if self.sorted {
                /*
                let mut current = self.head.as_mut();
                let mut current_val: Option<i32> = None;

                loop {
                    match current {
                        Some(node) => {

                            match &node.next {
                                Some(next) => {

                                    println!("Node = {}", next);
                                    current = node.next.as_mut();
                                },
                                None => {
                                    println!("Nothing");
                                }
                            }
                        },
                        None => {
                            break;
                        }
                    };
                }
                */

            } else {
                let mut current = self.head.as_mut();

                loop {
                    match current {
                        Some(node) => {
                            
                            match node.value {
                                Some(_val) => {
                                    current = node.next.as_mut();
                                },
                                None => {
                                    node.value = Some(value);
                                    break;
                                }
                            }
                        },
                        None => {
                            break;
                        }
                    };
                }
            }
        }
    }

    /*
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

            if self.sorted {
                loop {
                    match current {
                        Some(node) => {
                            // checks if the next node is None,
                            // if it is not None, set current to a &mut of the next node.
                            // if it is None, we are at the end of the list, 
                            // thus, set the next parameter to another node
                            
                            match &node.next {
                                Some(next) => {
                                    if next.value > value {
                                        /*
                                        let mut new_node = Box::new(SLLNode::new(value, None)); 
                                        new_node.next = node.next;
                                        println!("{} is larger than {}", next.value, value);
                                        */
                                        //let mut next_node = next;
                                        let mut new_node = Box::new(SLLNode::new(value, None));  
                                        new_node.next = Some(next.as_mut());
                                        node.next = Some(new_node);
                                        //new_node.next = current.unwrap();
                                        break;
                                    }
                                    current = node.next.as_mut();
                                },
                                None => {
                                    node.next = Some(Box::new(SLLNode::new(value, None)));
                                    self.num_elements += 1;
                                    break;
                                }
                            }
                            
                            /*
                            if node.next.is_some() {
                                current = node.next.as_mut();
                            } else {
                                node.next = Some(Box::new(SLLNode::new(value, None)));
                                self.num_elements += 1;
                                break;
                            }
                            */
                        },
                        None => {
                            break;
                        }
                    };
                }

                //let &mut n = current.unwrap();
                //println!("current = {}", n.value);

            } else {
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
    }
    */

    pub fn print_list(&self) {
        if self.head.is_some() {
            let mut current = &self.head;
            let mut position = 0;

            loop {
                match current {
                    Some(node) => {
                        println!("element {} = {}", position, node.to_string());
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