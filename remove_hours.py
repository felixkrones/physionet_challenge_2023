import os

source_dir = "data"
max_hours = 72

# For all files in all subfolders of the source_dir, remove them if they are older than max_hours. The hour is a string in the filename
for root, dirs, files in os.walk(source_dir):
    for filename in files:
        hour = int(filename[9:12])
        if hour > max_hours:
            os.remove(os.path.join(root, filename))
            print("Removed file: " + os.path.join(root, filename))
