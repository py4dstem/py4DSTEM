# `py4DSTEM.test` submodule

Files for testing py4DSTEM functionality using the pytest framework.


## Installation

with the pytest framework in mind - to run tests, please first
intall pytest with

`pip install -U pytest`



## Running tests

To run all tests, you can then do

`pytest`

from the command line - pytest will collect and run all the test
in this directory and its subdirectories. You can also run a
single test file or all files in a single test subdirectory with

`pytest test_file.py`
`pytest test_dir`



## Data and filepaths

Some tests need data files to run.  In order to avoid distibuting large
datasets with the package these files do not come pre-installed with the
package - you'll need to download them.  To do so, run the
`download_test_data.py` file in this directory, e.g. with

`python download_test_data.py`

There should now be a new `unit_test_data/` subdirectory containing the
necessary files for testing.



## Adding your own tests

When pytest is run it will find files in this directory and its
subdirectories with the format `test_*.py` or `*_test.py`.  Please
name your new file `test_*.py` for some short, descriptive `*`
specifying that nature of your tests.

Inside the file, any function called `test_*` will be found and run
by pytest.  In classes named `Test*`, any methods called `test_*` will
also be found and run.






