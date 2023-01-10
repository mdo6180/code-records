pub struct BSTNode<T> {
    pub value: T,
    pub left: Option<Box<BSTNode<T>>>,
    pub right: Option<Box<BSTNode<T>>>,
}

impl<T> BSTNode<T> {
    pub fn new(value: T) -> Self {
        BSTNode {
            value,
            left: None,
            right: None,
        }
    }

    pub fn left(mut self, node: BSTNode<T>) -> Self {
        self.left = Some(Box::new(node));
        self
    }

    pub fn right(mut self, node: BSTNode<T>) -> Self {
        self.right = Some(Box::new(node));
        self
    }
}