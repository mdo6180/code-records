class Sample:
    def __init__(self, name, path):
        self._name = name
        self._path = path
    
    @property
    def name(self):
        return self._name
    
    @property
    def path(self):
        return self._path

    @name.setter
    def name(self, name):
        print("setting name")
        self._name = name
    
    @path.setter
    def path(self, path):
        print("setting path")
        self._path = path

if __name__ == "__main__":
    sample1 = Sample("sample1", "path1")
    print(sample1.name)
    print(sample1.path)
    sample1.name = "sample2"
    sample1.path = "path2"
    print(sample1.name)
    print(sample1.path)