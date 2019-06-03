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
        'numpy >= 1.15, < 2.0',
        'scipy >= 1.1, < 1.2',
        'hyperspy >= 1.4, < 1.5',
        'PyQt5 >= 5.9, < 6',
        'pyqtgraph >= 0.10, < 0.11',
        'qtconsole >= 4.4, < 4.5'
    ])

