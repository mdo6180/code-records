import debugpy
import argparse



def attach_debugger():
    """
    Call this function in a file to enable debugging mode. It will listen for a debugger to attach on port 5678.

    Usage:
    Open a terminal and run the script with the `--debug` flag:
    ```
    $ python your_script.py --debug
    ```

    Then, in VS Code, open the debug tab, select "Python Debugger: Remote Attach" from the dropdown, and click on the play button. 
    The script will pause at breakpoints and you can inspect the values of variables in the debug console.
    """

    argument_parser = argparse.ArgumentParser(description="Validate a BagIt bag.")
    argument_parser.add_argument("-d", "--debug", action="store_true", help="Enable debugging mode.")
    args = argument_parser.parse_args()

    if args.debug:
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        print("Debugger attached")