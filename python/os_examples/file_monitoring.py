import os, time, hashlib


file = "hello.txt"

last_atime = os.stat(file).st_atime
last_mtime = os.stat(file).st_mtime
last_ctime = os.stat(file).st_ctime
last_hash = hashlib.sha256(open(file, "rb").read()).hexdigest()

while True:
    os_stat = os.stat(file)

    current_atime = os_stat.st_atime
    if current_atime != last_atime:
        # will print when manually open the file in editor
        print(f"File was accessed: {time.ctime(current_atime)}")    
        last_atime = current_atime

    current_mtime = os_stat.st_mtime
    if current_mtime != last_mtime:
        # will print when the file is saved in editor or in python script
        print(f"File was saved: {time.ctime(current_mtime)}")       
        last_mtime = current_mtime
    
        current_hash = hashlib.sha256(open(file, "rb").read()).hexdigest()
        if current_hash != last_hash:
            print(f"File content changed")
            last_hash = current_hash

    time.sleep(0.1)
