# https://docs.python.org/3/library/contextlib.html#using-a-context-manager-as-a-function-decorator

from contextlib import ContextDecorator



class mycontext(ContextDecorator):
    def __enter__(self):
        print('Starting')
        return self

    def __exit__(self, *exc):
        print('Finishing')
        return False


# Usage as a decorator
@mycontext()
def function():
    print('Inside function')

function()

# Usage as a context manager
print('--- before with statement ---')
with mycontext():
    print('Inside with statement')
print('--- after with statement ---')
