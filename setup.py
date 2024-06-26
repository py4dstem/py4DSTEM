from setuptools import setup, find_packages
from distutils.util import convert_path

with open("README.md", "r") as f:
    long_description = f.read()

version_ns = {}
vpath = convert_path("py4DSTEM/version.py")
with open(vpath) as version_file:
    exec(version_file.read(), version_ns)

setup(
    name="py4DSTEM",
    version=version_ns["__version__"],
    packages=find_packages(),
    description="An open source python package for processing and analysis of 4D STEM data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/py4dstem/py4DSTEM/",
    author="Benjamin H. Savitzky",
    author_email="ben.savitzky@gmail.com",
    license="GNU GPLv3",
    keywords="STEM 4DSTEM",
    python_requires=">=3.10",
    install_requires=[
        "numpy >= 1.19",
        "scipy >= 1.5.2",
        "h5py >= 3.2.0",
        "hdf5plugin >= 4.1.3",
        "ncempy >= 1.8.1",
        "matplotlib >= 3.2.2",
        "scikit-image >= 0.17.2",
        "scikit-learn >= 0.23.2, < 1.5",
        "scikit-optimize >= 0.9.0",
        "tqdm >= 4.46.1",
        "dill >= 0.3.3",
        "gdown >= 5.1.0",
        "dask >= 2.3.0",
        "distributed >= 2.3.0",
        "emdfile >= 0.0.14",
        "mpire >= 2.7.1",
        "threadpoolctl >= 3.1.0",
        "pylops >= 2.1.0",
        "colorspacious >= 1.1.2",
    ],
    extras_require={
        "ipyparallel": ["ipyparallel >= 6.2.4", "dill >= 0.3.3"],
        "cuda": ["cupy >= 10.0.0"],
        "acom": ["pymatgen >= 2022", "mp-api == 0.24.1"],
        "aiml": [
            "tensorflow <= 2.10.0",
            "tensorflow-addons <= 0.16.1",
            "crystal4D",
            "typeguard == 2.7",
        ],
        "aiml-cuda": [
            "tensorflow <= 2.10.0",
            "tensorflow-addons <= 0.16.1",
            "crystal4D",
            "cupy >= 10.0.0",
            "typeguard == 2.7",
        ],
        "numba": ["numba >= 0.49.1"],
    },
    package_data={
        "py4DSTEM": [
            "process/utils/scattering_factors.txt",
            "braggvectors/multicorr_row_kernel.cu",
            "braggvectors/multicorr_col_kernel.cu",
        ]
    },
)
