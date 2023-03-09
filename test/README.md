# `py4DSTEM.test` submodule

Testing py4DSTEM with pytest.



## Setup

Install the latest pytest with

`pip install -U pytest`


Some tests need data files to run.
In an environment with py4DSTEM installed,
do `python download_test_data.py` from this directory.
The script will make a new `unit_test_data` and
download the requisite files here.



## Running tests

To run all tests, you can then do `pytest` from
the command line - pytest will collect and run all the test
in this directory and its subdirectories. You can also run a
single test file or all files in a single test subdirectory with

`pytest test_file.py`
`pytest test_dir`





## Adding new tests

When pytest is run it will find files in this directory and its
subdirectories with the format `test_*.py` or `*_test.py`.
Name your new file `test_*.py` for some short, descriptive `*`
specifying that nature of your tests.

Inside the file, any function called `test_*` will be found and run
by pytest, and in classes named `Test*` any methods called `test_*` will
also be found and run.






