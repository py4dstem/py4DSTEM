import requests 
from typing import Union
import pathlib
import os


def download_file_from_google_drive(id_,destination, overwrite=False):
    """
    Downloads a file from google drive to the destination file path
    Args:
        id_ (str): File ID for the desired file, string of letters and numbers e.g.
            for https://drive.google.com/file/d/1bHv3u61Cr-y_GkdWHrJGh1lw2VKmt3UM/
            id='1bHv3u61Cr-y_GkdWHrJGh1lw2VKmt3UM'
        destination (Union[pathlib.PurePosixPath, pathlib.PureWindowsPath,str]): path
            file will be downloaded
        overwrite (bool): switch to add overwrite protection
    """

    if os.path.exists(destination) and overwrite == False:
        print("File Already Exists")
        return None
    
    elif os.path.exists(destination) and overwrite == True:
        print("File Already Existed, Downloading and Overwriting")
    else:
        print("Downloading {}".format(destination))

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
    
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id_ }, stream = True)
    token = get_confirm_token(response)
    
    if token:
        params = { 'id' : id_, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

    return None