mod linked_list;
//use linked_list::singly::{SinglyLinkedList, SLLNode};
use linked_list::singly::SinglyLinkedList;


fn main() {
    let mut list: SinglyLinkedList = SinglyLinkedList::new();
    //let node3: SLLNode = SLLNode::new(3, None);
    //let node2: SLLNode = SLLNode::new(2, Some(Box::new(node3)));
    //let node1: SLLNode = SLLNode::new(1, Some(Box::new(node2)));
    //list.head = Some(Box::new(node1));

    list.insert(1);
    list.insert(2);
    list.insert(3);
    list.insert(4);
    list.insert(5);
    list.insert(6);
    list.print_list();
}
