import json
import threading
import fcntl

class ThreadSafeJsonIO:
    def __init__(self, filename):
        self.filename = filename
        self.lock = threading.Lock()

    def read(self):
        with self.lock:
            with open(self.filename, 'r') as file:
                fcntl.flock(file.fileno(), fcntl.LOCK_SH)
                try:
                    return json.load(file)
                finally:
                    fcntl.flock(file.fileno(), fcntl.LOCK_UN)

    def write(self, data):
        with self.lock:
            with open(self.filename, 'w') as file:
                fcntl.flock(file.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(data, file, indent=4)
                finally:
                    fcntl.flock(file.fileno(), fcntl.LOCK_UN)

    def update(self, update_func):
        with self.lock:
            with open(self.filename, 'r+') as file:
                fcntl.flock(file.fileno(), fcntl.LOCK_EX)
                try:
                    data = json.load(file)
                    updated_data = update_func(data)
                    file.seek(0)
                    file.truncate()
                    json.dump(updated_data, file, indent=4)
                finally:
                    fcntl.flock(file.fileno(), fcntl.LOCK_UN)

# Usage example
json_io = ThreadSafeJsonIO('data.json')

# create the json file
new_data = {}
json_io.write(new_data)

def read_update_loop(threadId: int):
    i = 0

    while i < 50:
        # Reading
        data = json_io.read()

        # Updating
        def update_function(data):
            data['new_key'] = f'threadId_{threadId}'
            return data

        json_io.update(update_function)

        i += 1
