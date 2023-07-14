# A command line tool for downloading data to run the py4DSTEM test suite


import argparse
from os.path import join, exists
from os import makedirs

from py4DSTEM import _TESTPATH as testpath
from py4DSTEM.io import gdrive_download as download



# Make the argument parser
parser = argparse.ArgumentParser(
    description = "A command line tool for downloading data to run the py4DSTEM test suite"
)

# Set up data download options
data_options = [
    'tutorials',
    'io',
    'basic',
    'strain',
]

# Add arguments
parser.add_argument(
    "data",
    help = "which data to download.",
    choices = data_options,
)
parser.add_argument(
    "-o", "--overwrite",
    help = "if turned on, overwrite files that are already present. Otherwise, skips these files.",
    action = "store_true"
)
parser.add_argument(
    "-v", "--verbose",
    help = "turn on verbose output",
    action = "store_true"
)



# Get the command line arguments
args = parser.parse_args()


# Set up paths
if not exists(testpath):
    makedirs(testpath)


# Set data collection key
if args.data == 'tutorials':
    data = 'tutorials'
elif args.data == 'io':
    data = 'test_io'
elif args.data == 'basic':
    data = 'small_datacube'
elif args.data == 'strain':
    data = 'strain'
else:
    raise Exception(f"invalid data choice, {parser.data}")

# Download data
download(
    data,
    destination = testpath,
    overwrite = args.overwrite,
    verbose = args.verbose
)

# Always download the basic datacube
if args.data != 'basic':
    download(
        'small_datacube',
        destination = testpath,
        overwrite = args.overwrite,
        verbose = args.verbose
    )



