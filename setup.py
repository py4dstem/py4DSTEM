from setuptools import setup, find_packages
from distutils.util import convert_path

with open("README.md","r") as f:
    long_description = f.read()

version_ns = {}
vpath = convert_path('py4DSTEM/version.py')
with open(vpath) as version_file:
    exec(version_file.read(), version_ns)

setup(
    name='py4DSTEM',
    version=version_ns['__version__'],
    packages=find_packages(),
    description='An open source python package for processing and analysis of 4D STEM data.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/py4dstem/py4DSTEM/',
    author='Benjamin H. Savitzky',
    author_email='ben.savitzky@gmail.com',
    license='GNU GPLv3',
    keywords="STEM 4DSTEM",
    python_requires='>=3.7',
    install_requires=[
        'numpy >= 1.19',
        'scipy >= 1.5.2',
        'h5py >= 2.10.0',
        'ncempy >= 1.8.1',
        'matplotlib >= 3.4.2',
        'scikit-image >= 0.17.2',
        'scikit-learn >= 0.23.2',
        'PyQt5 >= 5.10',
        'pyqtgraph >= 0.11',
        'qtconsole >= 4.7.7',
        'ipywidgets >= 7.6.3',
        'tqdm >= 4.46.1',
        'dill >= 0.3.3',
        'gdown >= 4.4.0',
        'dask >= 2.3.0',
        'distributed >= 2.3.0'
        ],
    extras_require={
        'ipyparallel': ['ipyparallel >= 6.2.4', 'dill >= 0.3.3'],
        'cuda': ['cupy'],
        'acom': ['pymatgen'],
        'aiml': ['tensorflow == 2.4.1','tensorflow-addons','crystal4D'],
        'aiml-cuda': ['tensorflow-addons''crystal4D','cupy','numba >= 0.49.1'],
        'numba': ['numba >= 0.49.1']
        },
    entry_points={
        'console_scripts': ['py4DSTEM=py4DSTEM.gui.runGUI:launch']
    },
    package_data={
        'py4DSTEM':['process/utils/scattering_factors.txt']
    },
)
