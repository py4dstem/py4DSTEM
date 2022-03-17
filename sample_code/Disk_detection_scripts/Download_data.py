# This file will download the small dataset ~6.7 to the directory that it is run in

from py4DSTEM.io import download_file_from_google_drive


if __name__ == '__main__':
    download_file_from_google_drive("1EsbTlbbMZtIB9oZqEuo1jLwMU8o24soU", "./Ge_SiGe_ML_ideal.h5")
    print("file downloaded")