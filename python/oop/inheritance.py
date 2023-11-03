class Parent1:
    pass

class Parent2:
    pass

class Child(Parent1, Parent2):
    pass

class GrandChild(Child):
    pass

class GreatGrandChild(Child):
    pass

obj = GreatGrandChild()

print(isinstance(obj, Parent1)) # True
print(isinstance(obj, Parent2)) # True
