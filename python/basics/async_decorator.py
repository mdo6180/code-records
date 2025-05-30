import asyncio


# Simple decorator with parameters
def type_check_decorator(expected_type):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if all(isinstance(arg, expected_type) for arg in args):

                # Note: we do not use `await` here because we are not calling any other functions that are async other than the decorated function
                # this means that a regular decorator is sufficient for decorating an async function
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

        # Note: the wrapper function (i.e., the innermost function) must be async to handle async function calls but the outer decorator does not need to be async
        # Note: we have to define the wrapper function as async to use `await` inside it because of the asyncio.sleep call
        async def wrapper(*args, **kwargs):

            await asyncio.sleep(2)

            if all(isinstance(arg, expected_type) for arg in args):

                # Note: we must use `await` to call the decorated function
                return await func(*args, **kwargs)

            return "Invalid Input"

        return wrapper
    return decorator

# Example usage
# Note: the function being decorated must be async; this by definition implies that the decorated function can only be used in an async context
@async_type_check_decorator(int)
async def subtract_numbers(a, b):
    return a - b


async def main():
    # Note: we must use `await` to call the decorated function
    result = await add_numbers(5, 10)  # Should print 15
    print(result)

    result = await subtract_numbers(20, 30)  # Should print "Invalid Input"
    print(result)


asyncio.run(main())