import debugpy
import time


# 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to localhost.
debugpy.listen(5678)
print("Waiting for debugger attach")
debugpy.wait_for_client()

def main():
    while True:
        print('Hello1')
        time.sleep(2)
        print('Hello2')
        time.sleep(2)
        print('Hello3')
        time.sleep(2)
        print('Hello4')
        time.sleep(2)

main()


# To debug this script, run the script, then open the debug tab in vscode, 
# select "Python Debugger: Remote Attach" from the dropdown, then click on the play button.
# Add a breakpoint by clicking on the left side of the line number you want to break on.
# The script will pause at the breakpoint and you can inspect the values of variables in the debug console.

# Another way to add the breakpoints first, then run the script with the debugger attached