# py4DSTEM sample code and tutorials

This subdirectory contain sample code providing both an introduction to the py4DSTEM package and various examples of its use.
If you're new to the package, we recommend you start by running the Jupyter notebook Quickstart.ipynb.


There are two filetypes here: Jupyter notebooks (.ipynb) and python scripts (.py).
These filetypes both have advantages and disadvantages, and might be useful in different circumstances.
The .ipynb files are meant to be run interactively, and serve two use cases: (1) tutorials, with in-line discussion about what the code is doing and why, to help new users understand how and when to use which functions; and (2) interactive data analysis, which can be useful when a dataset benefits from live tuning of input parameters.
The interactivity may may the .ipynb files easier for novice programmers.
Some disadvantages to the .ipynb files is the necessity of direct user/programmer input at runtime, and possible ambiguity or introduction of error due to the possibility of running cells out of order.
The .py files are useful for fixed workflows which need no interactive input, and better ensure clarity and repeatibility by always running code in order as written.
They also lend themselves better to batch processing, allowing analysis of many datasets with a single command.


These two filetypes are not an either-or proposition.
In some cases, it may be beneficial to do some initial processing and parameter selection in an interactive Jupyter notebook, before moving to python scripts for more computationally intensive steps.


