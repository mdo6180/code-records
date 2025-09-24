class mycontext:
    def __enter__(self):
        print('Starting')
        return self

    def __exit__(self, *exc):
        print('Finishing')
        return False

# Usage as a context manager
print('--- before with statement ---')
with mycontext():
    print('Inside with statement')
print('--- after with statement ---')



# Context manager with arguments
class mycontext1:
    def __init__(self, arg=None):
        self.arg = arg

    def __enter__(self):
        print('Starting mycontext1')
        return self     # value bound to the `as` target

    def __exit__(self, *exc):
        print('Finishing mycontext1')
        return False

# Usage as a context manager
print('--- before with statement ---')
with mycontext1(arg='argument') as arg:
    print(f'Inside with statement with argument: {arg.arg}')
print('--- after with statement ---')


class mycontext2:
    def __init__(self, arg=None):
        self.arg = arg

    def __enter__(self):
        print('Starting mycontext2')
        return self.arg     # value bound to the `as` target

    def __exit__(self, *exc):
        print('Finishing mycontext2')
        return False

# Nested context managers
print('--- before with statement ---')
with mycontext1(arg='argument1') as arg1, mycontext2(arg='argument2') as arg2:
    print(f'Inside with statement with arguments: {arg1.arg}, {arg2}')
print('--- after with statement ---')

print('--- before with statement ---')
with mycontext1(arg='argument1') as arg1:
    with mycontext2(arg='argument2') as arg2:
        print(f'Inside with statement with arguments: {arg1.arg}, {arg2}')
print('--- after with statement ---')