from datetime import date
from pathlib import Path
import bagit



bag_path = Path("./mydata")

# load the bag
bag = bagit.Bag(bag_path)

# update bag info metadata
bag.info['Internal-Sender-Description'] = f'Updated on {date.today()}.'
bag.info['Authors'] = ['Minh Quan Do', 'John Kunze']
bag.save()