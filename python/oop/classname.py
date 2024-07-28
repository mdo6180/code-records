class A:
    def __init__(self) -> None:
        print("class A init")

class B:
    def __init__(self, classname) -> None:
        print("class B init")

        classA = classname()

b = B(classname=A)
