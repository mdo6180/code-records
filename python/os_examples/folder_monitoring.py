import os, time, shutil



folder = "monitor_folder"
if not os.path.exists(folder):
    os.makedirs(folder)
else:
    shutil.rmtree(folder)
    os.makedirs(folder)



last_atime = os.stat(folder).st_atime
last_mtime = os.stat(folder).st_mtime
last_ctime = os.stat(folder).st_ctime

while True:
    time.sleep(0.1)

    for root, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            print(f"File: {os.path.join(root, filename)}")

    current_atime = os.stat(folder).st_atime
    if current_atime != last_atime:
        print(f"Folder was accessed: {time.ctime(current_atime)}")
        last_atime = current_atime

    current_mtime = os.stat(folder).st_mtime
    if current_mtime != last_mtime:
        print(f"Folder was modified: {time.ctime(current_mtime)}")
        last_mtime = current_mtime
    
    current_ctime = os.stat(folder).st_ctime
    if current_ctime != last_ctime:
        print(f"Folder metadata changed: {time.ctime(current_ctime)}")
        last_ctime = current_ctime
