import bagit
from pathlib import Path
import debugpy
import argparse


argument_parser = argparse.ArgumentParser(description="Validate a BagIt bag.")
argument_parser.add_argument("-d", "--debug", action="store_true", help="Enable debugging mode.")
args = argument_parser.parse_args()

if args.debug:
    debugpy.listen(5678)
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
    print("Debugger attached")


bag_path = Path("./mydata")

bag = bagit.Bag(bag_path)
if bag.is_valid():
    print("yay :)")
else:
    print("boo :(")