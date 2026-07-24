from pathlib import Path
import bagit


data_folder = Path("./mydata")

# take the contents of the mydata folder and create a bag in the same location
bag = bagit.make_bag(data_folder, {'Contact-Name': 'John Kunze'})