mod linked_list;
use linked_list::singly::{SinglyLinkedList, SLLNode};
//use linked_list::singly::SinglyLinkedList;


fn main() {
    let node3: SLLNode = SLLNode::new(Some(3), None);
    let node2: SLLNode = SLLNode::new(Some(2), Some(Box::new(node3)));
    let node1: SLLNode = SLLNode::new(Some(1), Some(Box::new(node2)));
    let mut list: SinglyLinkedList = SinglyLinkedList::new(Some(Box::new(node1)), true);
    //let mut list: SinglyLinkedList = SinglyLinkedList::new(None, true);

    list.insert(1);
    list.print_list();

    /*
    list.insert(1);
    list.insert(2);
    list.insert(5);
    list.insert(6);
    list.insert(3);
    list.insert(4);
    list.print_list();
    */
}
