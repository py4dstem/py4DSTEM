# This file downloads test files to enable running py4DSTEM unit
# tests with the pytest framework.


from py4DSTEM import _TESTPATH
#from os.path import join
filepath = _TESTPATH


if __name__ == '__main__':

    from py4DSTEM.io import download_file_from_google_drive as download

    download(
        id_ = 'unit_test_data',
        destination = filepath,
        overwrite = True
    )
    #print(filepath)

