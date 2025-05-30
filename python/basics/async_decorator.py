import asyncio


# Simple decorator with parameters
def type_check_decorator(expected_type):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if all(isinstance(arg, expected_type) for arg in args):
                return func(*args, **kwargs)
            return "Invalid Input"
        return wrapper
    return decorator

# Example usage
@type_check_decorator(int)
async def add_numbers(a, b):
    return a + b


# Simple decorator with parameters that call async functions (e.g. with asyncio.sleep)
def async_type_check_decorator(expected_type):
    def decorator(func):
        async def wrapper(*args, **kwargs):

            await asyncio.sleep(2)

            if all(isinstance(arg, expected_type) for arg in args):
                return await func(*args, **kwargs)

            return "Invalid Input"

        return wrapper
    return decorator

# Example usage
@async_type_check_decorator(int)
async def subtract_numbers(a, b):
    return a - b


async def main():
    result = await add_numbers(5, 10)  # Should print 15
    print(result)

    result = await subtract_numbers(20, 30)  # Should print "Invalid Input"
    print(result)


asyncio.run(main())