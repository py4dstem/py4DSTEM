import py4DSTEM

load_file = False
key = 'https://drive.google.com/file/d/1-HUd5dwicmj2kwZzsUmcS_v3b7qf8r3b/view?usp=sharing'
filepath = "/media/AuxDriveB/Data/TaraMishra/test_file.mib"

if __name__ == '__main__':

    if load_file:
        print("loading test .mib file...")
        py4DSTEM.io.download_file_from_google_drive(key, filepath)
        print("done.")
        print()

    print("reading .mib file...")
    datacube = py4DSTEM.import_file(filepath)
    print("done.")
    print()

    print("loaded data:")
    print(datacube)
    print()

