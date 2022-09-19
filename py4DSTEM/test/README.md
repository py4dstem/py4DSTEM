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
package - you'll need to download them.  To do so, open the file
`download_test_data.py` and update the variable `filepath` to point to this
directory (`py4DSTEM/test/`) on your local installation. Then run
`download_test_data.py` in an environment that has py4DSTEM installed,
e.g. with

`python download_test_data.py`

There should now be a new `unit_test_data/` subdirectory containing the
necessary files for testing.



## Adding your own tests

When pytest is run it will find files in this directory and its
subdirectories with the format `test_*.py` or `*_test.py`.  Please
name your new file `test_*.py` for some short, descriptive `*`
specifying that nature of your tests.

Inside the file, any function called `test_*` will be found and run
by pytest.

If multiple tests need the same boilerplate code to be run to set up the
test, these functions can be placed inside a class named `Test*`, the
boilerplate code can be placed in the `__init__` function, and the
tests can be written as individual methods named `test_*`. pytest will
find and run each of these.  Note that a new instance of the class will
be created for each test.





