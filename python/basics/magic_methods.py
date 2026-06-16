class Dictionary:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __contains__(self, key):
        return key in self.data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __repr__(self):
        return f"Dictionary({self.data})"


dictionary = Dictionary({'a': 1, 'b': 2, 'c': 3})
print(dictionary['a'])  # Output: 1
dictionary['d'] = 4
print(dictionary)  # Output: Dictionary({'a': 1, 'b': 2, 'c': 3, 'd': 4})
del dictionary['b']
print(dictionary)  # Output: Dictionary({'a': 1, 'c': 3, 'd': 4})
print('c' in dictionary)  # Output: True
print(len(dictionary))  # Output: 3