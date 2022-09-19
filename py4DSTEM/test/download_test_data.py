# This file downloads test files to enable running py4DSTEM unit
# tests with the pytest framework.

# Please set the filepath below to point to the `py4DSTEM/test/`
# directory on your local installation. Then run this file to
# download test data.

filepath = '/home/ben/projects/self/py4DSTEM/py4DSTEM/py4DSTEM/test/'


if __name__ == '__main__':

    from py4DSTEM.io import download_file_from_google_drive as download

    download(
        id_ = 'unit_test_data',
        destination = filepath,
        overwrite = True
    )


