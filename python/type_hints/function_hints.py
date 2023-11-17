from typing import List, Dict, Union


numbers_list = [1,2,3]


def func(numbers: Union[int, List[int]]):
    if isinstance(numbers, int):
        numbers_list.append(numbers)
    elif isinstance(numbers, list):
        numbers_list.extend(numbers)
    else:
        raise TypeError(f"'numbers' must be of either type int or list, not {type(numbers)}") 


if __name__ == "__main__":
    func(4)
    print(numbers_list)

    func([5,6,7])
    print(numbers_list)
