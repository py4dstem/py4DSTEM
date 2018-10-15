from setuptools import setup, find_packages

exec(open('version.py').read()) # Reads the current __version__

setup(name='py4DSTEM',
    version=__version__,
    packages=find_packages(),
    description='Open source processing and analysis of 4D STEM data.',
    url='https://github.com/bsavitzky/py4DSTEM/',
    author='Benjamin H. Savitzky',
    author_email='ben.savitzky@gmail.com',
    license='GNU GPLv3',
    keywords="STEM 4DSTEM")

