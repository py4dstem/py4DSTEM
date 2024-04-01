"""
Script to update the patch version number of the py4DSTEM package.
"""

version_file_path = "py4DSTEM/version.py"

with open(version_file_path, "r") as f:
    lines = f.readlines()

line_split = lines[0].split(".")
patch_number = line_split[2].split("'")[0].split('"')[0]

# Increment patch number
patch_number = str(int(patch_number) + 1) + '"\n'


new_line = line_split[0] + "." + line_split[1] + "." + patch_number

with open(version_file_path, "w") as f:
    f.write(new_line)
