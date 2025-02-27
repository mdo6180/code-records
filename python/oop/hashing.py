class Hello:
    def __init__(self, name: str) -> None:
        self.name = f"{name}-{hash(self)}"

    def hello(self):
        return "Hello, World!"


if __name__ == "__main__":
    hello1 = Hello("World")
    print(hello1.name)

    hello2 = Hello("World")
    print(hello2.name)

    hello3 = Hello("World")
    print(hello3.name)