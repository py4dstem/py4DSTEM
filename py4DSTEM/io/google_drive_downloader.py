import gdown
from typing import Union
import pathlib
import os


# Built-in sample datasets

sample_data_ids = {
    'FCUNet' : 'TODO-add url/id',
    'sample_diffraction_pattern':'1ymYMnuDC0KV6dqduxe2O1qafgSd0jjnU'
}


def download_file_from_google_drive(id_,destination, overwrite=False):
    """
    Downloads a file from google drive to the destination file path.

    Args:
        id_ (str): File ID for the desired file.  May be:
            - the file id, i.e. for
              https://drive.google.com/file/d/1bHv3u61Cr-y_GkdWHrJGh1lw2VKmt3UM/
              id='1bHv3u61Cr-y_GkdWHrJGh1lw2VKmt3UM'
            - the complete URL,
            - a special string denoting a sample dataset. For a list of sample
              datasets and their keys, run get_sample_data_ids().
        destination (Union[pathlib.PurePosixPath, pathlib.PureWindowsPath,str]): path
            file will be downloaded
        overwrite (bool): switch to add or remove overwrite protection
    """

    # Check if a file at the destination filepath exists
    if os.path.exists(destination) and overwrite == False:
        print("File already exists")
        return None
    elif os.path.exists(destination) and overwrite == True:
        print(f"File already existed, downloading and overwriting")
        os.remove(destination)

    # Check if the id_ is a key pointing to a known sample dataset
    if id_ in sample_data_ids.keys():
        id_ = sample_data_ids[id_]

    # Check if `id_` is a file ID or a URL, and download
    if id_[:4].lower()=='http':
        gdown.download(url=id_,output=destination,fuzzy=True)
    else:
        gdown.download(id=id_,output=destination,fuzzy=True)

    return None


def get_sample_data_ids():
    return sample_data_ids.keys()



