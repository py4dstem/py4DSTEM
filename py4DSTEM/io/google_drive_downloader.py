import gdown
import os

from py4DSTEM.io.sample_file_ids import file_ids, collection_ids

def gdrive_download(
    id_,
    destination = None,
    overwrite = False,
    filename = None,
    verbose = True,
    ):
    """
    Downloads a file or collection of files from google drive.

    Parameters
    ----------
    id_ : str
        File ID for the desired file.  May be either a key from the list
        of files and collections of files accessible at get_sample_file_ids(),
        or a complete url, or the portions of a google drive link specifying
        the file ID, i.e. for the address
        https://drive.google.com/file/d/1bHv3u61Cr-y_GkdWHrJGh1lw2VKmt3UM/,
        the id string '1bHv3u61Cr-y_GkdWHrJGh1lw2VKmt3UM'.
    destination : None or str
        The location to download to. If None, downloads to the current
        working directory. If a string is passed which points to a valid
        location on the filesystem, downloads there. For collections of
        files, makes a new directory named after the collection id and
        places all files there.
    overwrite : bool
        Turns overwrite protection on/off.
    filename : None or str
        Used only if `id_` is a url or gdrive id. Specifies the name of the
        output file.  If left as None, saves to 'gdrivedownload.file'. If `id_`
        is a key from the sample file id list, this parameter is ignored and
        the filenames in that list are used.
    verbose : bool
        Toggles verbose output
    """
    # parse destination
    if destination is None:
        destination = os.getcwd()
    assert(os.path.exists(destination)), f"`destination` must exist on filesystem. Received {destination}"

    # download single files
    if id_ not in collection_ids:

        # assign the name and id
        kwargs = {
            'fuzzy' : True
        }
        if id_ in file_ids:
            f = file_ids[id_]
            filename = f[0]
            kwargs['id'] = f[1]
        else:
            filename = 'gdrivedownload.file' if filename is None else filename
            kwargs['url'] = id_

        # download
        kwargs['output'] = os.path.join(destination, filename)
        if not(overwrite) and os.path.exists(kwargs['output']):
            if verbose:
                print(f"A file already exists at {kwargs['output']}, skipping...")
        else:
            gdown.download( **kwargs )

    # download a collections of files
    else:

        # set destination
        destination = os.path.join(destination, id_)
        if not os.path.exists(destination):
            os.mkdir(destination)

        # loop
        for x in collection_ids[id_]:
            file_name,file_id = file_ids[x]
            output = os.path.join(destination, file_name)
            # download
            if not(overwrite) and os.path.exists(output):
                if verbose:
                    print(f"A file already exists at {output}, skipping...")
            else:
                gdown.download(
                    id = file_id,
                    output = output,
                    fuzzy = True
                )

