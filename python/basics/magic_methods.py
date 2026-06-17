from typing import List, Tuple, Union



class Matrix:
    def __init__(self, data: List[List[int]]):
        self.data = data

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.data[index]

        elif isinstance(index, tuple):
            row, col = index

            # matrix[0:3, 2]
            if isinstance(row, slice) and isinstance(col, int):
                return [r[col] for r in self.data[row]]

            # matrix[1, 2]
            elif isinstance(row, int) and isinstance(col, int):
                return self.data[row][col]


matrix = Matrix(
    [[1, 2, 3], 
     [4, 5, 6], 
     [7, 8, 9]]
)
print(matrix[0])
print(matrix[0, 1])
print(matrix[0:3, 2])


# Note: for Anacostia, we can use the __getitem__ method like so: stream[hash, "location"] to get the location associated with the hash key in the stream.
# We can also do stream[hash, "content"] to get the content associated with the hash key in the stream. 
# This allows us to easily access specific pieces of information from the stream using a simple and intuitive syntax.