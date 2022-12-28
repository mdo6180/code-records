import numpy as np
import os


# example shows what happens when an array is initialized to dtype=int
# reassignment to float64 fails because float64 array is getting rounded to 0

if __name__ == "__main__":
    zeros_array = np.zeros((3, 3, 3))
    new_array = np.random.rand(3, 3)
    zeros_array[0] = new_array
    print(zeros_array)

    zeros_array = np.zeros((3, 3, 3), dtype=int)
    new_array = np.random.rand(3, 3)
    zeros_array[0] = new_array
    print(zeros_array)
