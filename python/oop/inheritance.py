class Parent:
    pass

class Child(Parent):
    pass

class GrandChild(Child):
    pass

class GreatGrandChild(Child):
    pass

obj = GreatGrandChild()

print(isinstance(obj, Parent)) # True
