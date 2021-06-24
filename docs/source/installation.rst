.. _installation:

Installation
============

The recommended installation for py4DSTEM uses the `Anaconda <https://www.anaconda.com/>`_ Python distribution.
First, download and install Anaconda. Instructions can be found `here <http://www.anaconda.com/download>`_.
Then open a terminal and run::

    conda update conda
    conda create -n py4dstem python==3.8
    conda activate py4dstem
    conda install pip
    pip install py4dstem

If you're running Windows, you should then also run::

    conda install pywin32

In order, these commands

* ensure your installation of anaconda is up-to-date
* make a virtual environment (see below)
* enter the environment
* make sure your new environment talks nicely to `pip <https://pypi.org/project/pip>`_, a tool for installing Python packages
* use pip to install py4DSTEM
* on Windows: enable python to talk to the windows API


.. _virtualenvironments:

.. Attention:: **Virtual environments**

   A Python virtual environment is its own siloed version of Python, with its own set of packages and modules, kept separate from any other Python installations on your system.
   In the instructions above, we created a virtual environment to make sure packages that have different dependencies don't conflict with one another.
   For instance, as of this writing, some of the scientific Python packages don't work well with Python 3.9 - but you might have some other applications on your computer that *need* Python 3.9.
   Using virtual environments solves this problem.
   In this example, we're creating and navigating virtual environments using Anaconda.

   Because these directions install py4DSTEM to its own virtual environment, each time you want to use py4DSTEM, you'll need to activate this environment.
   
   * In the command line, you can do this with ``conda activate py4dstem``.
   * In the Anaconda Navigator, you can do this by clicking on the Environments tab and then clicking on ``py4dstem``.



