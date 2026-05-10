import os

for filename in os.listdir('.'):
    if filename.endswith('-std.rttm'):
        new_name = filename[:-9] + '-avd.rttm'
        os.rename(filename, new_name)
        print(f'Renamed: {filename} -> {new_name}')