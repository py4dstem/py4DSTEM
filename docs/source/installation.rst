.. _installation:

Installation
============

.. contents:: Table of Contents
    :depth: 4



Setting up Python
-----------------

The recommended installation for py4DSTEM uses the `Anaconda <https://www.anaconda.com/>`_ Python distribution. Alternatives such as  `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_, `Mamba <https://mamba.readthedocs.io/en/latest/>`_, `pip virtualenv <https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/>`_, and `poetry <https://python-poetry.org>`_ will work, but here we assume the use of Anaconda. See :ref:`virtualenvironments`, for more details. 
The instructions to download and install Anaconda can be found `here <http://www.anaconda.com/download>`_.




.. The overview of installation process is: 

.. * make a virtual environment (see below)
.. * enter the environment
.. * install py4DSTEM

Recommended Installation
------------------

There are three ways to install py4DSTEM:

#. Anaconda (miniconda / mamba)
#. Pip
#. Installing from Source 

The easiest way to install py4DSTEM is to use the pre packaged anaconda version. This is an overview of what the installation process looks like, for OS specific instructions see below.

Anaconda
********

Windows
^^^^^^^

.. code-block:: shell
    :linenos:
    :caption: Windows base install

    conda create -n py4dstem python=3.9
    conda activate py4dstem
    conda install -c conda-forge py4dstem
    conda install -c conda-forge pywin32
    # optional but recomended 
    conda install jupyterlab pymatgen

Linux
^^^^^

.. code-block:: shell
    :linenos:
    :caption: Linux base install

    
    conda create -n py4dstem python=3.9
    conda activate py4dstem
    conda install -c conda-forge py4dstem
    # optional but recomended 
    conda install jupyterlab pymatgen

Mac (Intel)
^^^^^^^^^^^
.. code-block:: shell
    :linenos:
    :caption: Intel Mac base install

    
    conda create -n py4dstem python=3.9
    conda activate py4dstem
    conda install -c conda-forge py4dstem
    # optional but recomended 
    conda install jupyterlab pymatgen

Mac (Apple Silicon M1/M2)
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell
    :linenos:
    :caption: Apple Silicon Mac base install


    conda create -n py4dstem python=3.9
    conda activate py4dstem
    conda install pyqt hdf5
    conda install -c conda-forge py4dstem
    # optional but recomended 
    conda install jupyterlab pymatgen


Advanced Installation
---------------------

Installing optional dependencies:
*********************************

Some of the features and modules require extra dependencies which can easily be installed using either Anaconda or Pip.

Anaconda
********

Windows
^^^^^^^

.. code-block:: shell
    :linenos:
    :caption: Windows Anaconda install ACOM

    conda create -n py4dstem python=3.9
    conda activate py4dstem
    conda install -c conda-forge py4dstem pymatgen
    conda install -c conda-forge pywin32

Running py4DSTEM code with GPU acceleration requires an NVIDIA GPU (AMD has beta support but hasn't been tested) and Nvidia Drivers installed on the system. 

.. code-block:: shell
    :linenos:
    :caption: Windows Anaconda install GPU 

    conda create -n py4dstem python=3.9
    conda activate py4dstem
    conda install -c conda-forge py4dstem cupy cudatoolkit
    conda install -c conda-forge pywin32


If you are looking to run the ML-AI features you are required to install tensorflow, this can be done with CPU only and GPU support. 

.. code-block:: shell
    :linenos:
    :caption: Windows Anaconda install ML-AI CPU 

    conda create -n py4dstem python=3.9
    conda activate py4dstem
    conda install -c conda-forge py4dstem 
    pip install tensorflow==2.4.1 tensorflow-addons<=0.14 crystal4D
    conda install -c conda-forge pywin32

.. code-block:: shell
    :linenos:
    :caption: Windows Anaconda install ML-AI GPU 

    conda create -n py4dstem python=3.9
    conda activate py4dstem
    conda install -c conda-forge py4dstem 
    conda install -c conda-forge cupy cudatoolkit=11.0
    pip install tensorflow==2.4.1 tensorflow-addons<=0.14 crystal4D
    conda install -c conda-forge pywin32



Linux
^^^^^

.. code-block:: shell
    :linenos:
    :caption: Linux Anaconda install ACOM

    conda create -n py4dstem python=3.9
    conda activate py4dstem
    conda install -c conda-forge py4dstem pymatgen

Running py4DSTEM code with GPU acceleration requires an NVIDIA GPU (AMD has beta support but hasn't been tested) and Nvidia Drivers installed on the system. 

.. code-block:: shell
    :linenos:
    :caption: Linux Anaconda install GPU 

    conda create -n py4dstem python=3.9
    conda activate py4dstem
    conda install -c conda-forge py4dstem cupy cudatoolkit


If you are looking to run the ML-AI features you are required to install tensorflow, this can be done with CPU only and GPU support. 

.. code-block:: shell
    :linenos:
    :caption: Linux Anaconda install ML-AI CPU 

    conda create -n py4dstem python=3.9
    conda activate py4dstem
    conda install -c conda-forge py4dstem 
    pip install tensorflow==2.4.1 tensorflow-addons<=0.14 crystal4D

.. code-block:: shell
    :linenos:
    :caption: Linux Anaconda install ML-AI GPU 

    conda create -n py4dstem python=3.9
    conda activate py4dstem
    conda install -c conda-forge py4dstem 
    conda install -c conda-forge cupy cudatoolkit=11.0
    pip install tensorflow==2.4.1 tensorflow-addons<=0.14 crystal4D



Mac (Intel)
^^^^^^^^^^^
.. code-block:: shell
    :linenos:
    :caption: Intel Mac Anaconda install ACOM

    conda create -n py4dstem python=3.9
    conda activate py4dstem
    conda install -c conda-forge py4dstem pymatgen


Tensorflow does not support AMD GPUs so while ML-AI features can be run on an Intel Mac they are not GPU accelerated  

.. code-block:: shell
    :linenos:
    :caption: Intel Mac Anaconda install ML-AI CPU 

    conda create -n py4dstem python=3.9
    conda activate py4dstem
    conda install -c conda-forge py4dstem 
    pip install tensorflow==2.4.1 tensorflow-addons<=0.14 crystal4D

Mac (Apple Silicon M1/M2)
^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: shell
    :linenos:
    :caption: Apple Silicon Mac Anaconda install ACOM

    conda create -n py4dstem python=3.9
    conda activate py4dstem
    conda install -c conda-forge py4dstem pymatgen



Tensorflow's support of Apple silicon GPUs is limited, and while there are steps that should enable GPU acceleration they have not been tested, but CPU only has been tested. 

.. code-block:: shell
    :linenos:
    :caption: Apple Silicon Mac Anaconda install ML-AI CPU 

    conda create -n py4dstem python=3.9
    conda activate py4dstem
    conda install -c conda-forge py4dstem 
    pip install tensorflow==2.4.1 tensorflow-addons<=0.14 crystal4D

.. Attention:: **GPU Accelerated Tensorflow on Apple Silicon**

    This is an untested install method and it may not work. If you try and face issues please post an issue on `github <https://github.com/py4dstem/py4DSTEM/issues>`_.


.. code-block:: shell
    :linenos:
    :caption: Apple Silicon Mac Anaconda install ML-AI GPU

    
    conda create -n py4dstem python=3.9
    conda activate py4dstem
    conda install -c apple tensorflow-deps
    pip install tensorflow-macos==2.5.0 tensorflow-addons<=0.14 crystal4D tensorflow-metal
    conda install -c conda-forge py4dstem 



Pip
***

Windows
^^^^^^^

.. code-block:: shell
    :linenos:
    :caption: Windows pip install ACOM

    conda create -n py4dstem python=3.9
    conda activate py4dstem
    pip install py4dstem[acom] 
    conda install -c conda-forge pywin32

Running py4DSTEM code with GPU acceleration requires an NVIDIA GPU (AMD has beta support but hasn't been tested) and Nvidia Drivers installed on the system. 

.. code-block:: shell
    :linenos:
    :caption: Windows pip install GPU 

    conda create -n py4dstem python=3.9
    conda activate py4dstem
    pip install py4dstem[cuda]
    conda install -c conda-forge pywin32


If you are looking to run the ML-AI features you are required to install tensorflow, this can be done with CPU only and GPU support. 

.. code-block:: shell
    :linenos:
    :caption: Windows pip install ML-AI CPU 

    conda create -n py4dstem python=3.9
    conda activate py4dstem
    pip install py4dstem[aiml]
    conda install -c conda-forge pywin32

.. code-block:: shell
    :linenos:
    :caption: Windows pip install ML-AI GPU 

    conda create -n py4dstem python=3.9
    conda activate py4dstem
    conda install -c conda-forge cudatoolkit=11.0
    pip install py4dstem[aiml-cuda]
    conda install -c conda-forge pywin32

Linux
^^^^^

.. code-block:: shell
    :linenos:
    :caption: Linux pip install ACOM

    conda create -n py4dstem python=3.9
    conda activate py4dstem
    pip install py4dstem[acom] 

Running py4DSTEM code with GPU acceleration requires an NVIDIA GPU (AMD has beta support but hasn't been tested) and Nvidia Drivers installed on the system. 

.. code-block:: shell
    :linenos:
    :caption: Linux pip install GPU 

    conda create -n py4dstem python=3.9
    conda activate py4dstem
    pip install py4dstem[cuda]


If you are looking to run the ML-AI features you are required to install tensorflow, this can be done with CPU only and GPU support. 

.. code-block:: shell
    :linenos:
    :caption: Linux pip install ML-AI CPU 

    conda create -n py4dstem python=3.9
    conda activate py4dstem
    pip install py4dstem[aiml]

.. code-block:: shell
    :linenos:
    :caption: Linux pip install ML-AI GPU 

    conda create -n py4dstem python=3.9
    conda activate py4dstem
    conda install -c conda-forge cudatoolkit=11.0
    pip install py4dstem[aiml-cuda]

Mac (Intel)
^^^^^^^^^^^
.. code-block:: shell
    :linenos:
    :caption: Intel Mac pip install ACOM

    conda create -n py4dstem python=3.9
    conda activate py4dstem
    pip install py4dstem[acom]


Tensorflow does not support AMD GPUs so while ML-AI features can be run on an Intel Mac they are not GPU accelerated  

.. code-block:: shell
    :linenos:
    :caption: Intel Mac pip install ML-AI CPU 

    conda create -n py4dstem python=3.9
    conda activate py4dstem
    pip install py4dstem[aiml]

Mac (Apple Silicon M1/M2)
^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: shell
    :linenos:
    :caption: Apple Silicon Mac pip install ACOM

    conda create -n py4dstem python=3.9
    conda activate py4dstem
    pip install py4dstem[acom]
    conda install -c conda-forge py4dstem pymatgen



Tensorflow's support of Apple silicon GPUs is limited, and while there are steps that should enable GPU acceleration they have not been tested, but CPU only has been tested. 

.. code-block:: shell
    :linenos:
    :caption: Apple Silicon Mac Anaconda install ML-AI CPU 

    conda create -n py4dstem python=3.9
    conda activate py4dstem
    pip install py4dstem[aiml]

.. Attention:: **GPU Accelerated Tensorflow on Apple Silicon**

    This is an untested install method and it may not work. If you try and face issues please post an issue on `github <https://github.com/py4dstem/py4DSTEM/issues>`_.


.. code-block:: shell
    :linenos:
    :caption: Apple Silicon Mac Anaconda install ML-AI GPU

    
    conda create -n py4dstem python=3.9
    conda activate py4dstem
    conda install -c apple tensorflow-deps
    pip install tensorflow-macos==2.5.0 tensorflow-addons<=0.14 crystal4D tensorflow-metal py4dstem 


Installing from Source 
******************

To checkout the latest bleeding edge features, or contriubte your own features you'll need to install py4DSTEM from source. Luckily this is easy and can be done by simply running:

.. code-block:: shell
    :linenos:

    git clone 
    git checkout <branch> # e.g. git checkout dev
    pip install -e . 

Alternatively, you can try single step method:

.. code-block:: shell
    :linenos:

    pip install git+https://github.com/py4DSTEM/py4DSTEM.git@dev # install the dev branch


Docker
******

Overview 
^^^^^^^^
    "Docker is an open platform for developing, shipping, and running applications. Docker enables you to separate your applications from your infrastructure so you can deliver software quickly. With Docker, you can manage your infrastructure in the same ways you manage your applications. By taking advantage of Docker’s methodologies for shipping, testing, and deploying code quickly, you can significantly reduce the delay between writing code and running it in production." 
    c.f. `Docker website <https://docs.docker.com/get-started/overview/>`_

Installation
^^^^^^^^^^^^

There are py4DSTEM Docker images available on dockerhub, which can be pulled and run or built upon. Checkout the dockerhub repository to see all the versions aviale or simply run the below to get the latest version.
While Docker is extremely powerful and aims to greatly simplify depolying software, it is also a complex and nuanced topic. If you are interested in using it, and are having troubles getting it to work please file an issue on the github. 
To use Docker you'll first need to `install Docker <https://docs.docker.com/engine/install/>`_. After which you can run the images with the following commands.

.. code-block:: shell
    :linenos:

    docker pull arakowsk/py4dstem:latest
    docker run <Docker options> py4dstem:latest <commands> <args>

Alternatively, you can use `Docker Desktop <https://www.docker.com/products/docker-desktop/>`_ which is a GUI interface for Docker and may be an easier method for running the images for less experienced users. 


Troubleshooting
---------------

If you face any issues, see the common errors below, and if there's no solution please file an issue on the `git repository <https://github.com/py4dstem/py4DSTEM/issues>`_.

Some common errors: 
- make sure you've activated the right environment
- when installing subsections sometimes the quotation marks can be tricky dpeending on os, terminal etc. 
- GPU drivers - tricky to explain 






.. Attention:: **Virtual environments**
.. _virtualenvironments:

A Python virtual environment is its own siloed version of Python, with its own set of packages and modules, kept separate from any other Python installations on your system.
In the instructions above, we created a virtual environment to make sure packages that have different dependencies don't conflict with one another.
For instance, as of this writing, some of the scientific Python packages don't work well with Python 3.9 - but you might have some other applications on your computer that *need* Python 3.9.
Using virtual environments solves this problem.
In this example, we're creating and navigating virtual environments using Anaconda.

Because these directions install py4DSTEM to its own virtual environment, each time you want to use py4DSTEM, you'll need to activate this environment.

* In the command line, you can do this with ``conda activate py4dstem``.
* In the Anaconda Navigator, you can do this by clicking on the Environments tab and then clicking on ``py4dstem``.