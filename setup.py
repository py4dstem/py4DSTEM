from setuptools import setup, find_packages

exec(open('py4DSTEM/version.py').read()) # Reads the current __version__

setup(name='py4DSTEM',
    version=__version__,
    packages=find_packages(),
    description='Open source processing and analysis of 4D STEM data.',
    url='https://github.com/bsavitzky/py4DSTEM/',
    author='Benjamin H. Savitzky',
    author_email='ben.savitzky@gmail.com',
    license='GNU GPLv3',
    keywords="STEM 4DSTEM",
    install_requires=[
        'numpy >= 1.15',
        'scipy >= 1.1',
        'hyperspy >= 1.4',
        'PyQt5 >= 5.9, < 6',
        'pyqtgraph >= 0.10, < 0.11',
        'qtconsole >= 4.4, < 4.5',
        'ncempy >= 1.4.2',
        'tqdm',
        'ipywidgets',
        'scikit-learn'
        ],
    extras_require={
        'ipyparallel': ['ipyparallel >= 6.2.4'],
        'dask': ['dask >= 2.3.0', 'distributed >= 2.3.0']
        },
    entry_points= {
        'console_scripts': ['py4DSTEM=py4DSTEM.gui.runGUI:launch']
    })
