# When run as a Python script, this file
# makes a folder called 'unit_test_data' if one
# doesn't already exist, and downloads
# py4DSTEM's test data there.


from py4DSTEM import _TESTPATH

filepath = _TESTPATH


if __name__ == "__main__":
    from py4DSTEM.io import download_file_from_google_drive as download

    download(id_="unit_test_data", destination=filepath, overwrite=True)
