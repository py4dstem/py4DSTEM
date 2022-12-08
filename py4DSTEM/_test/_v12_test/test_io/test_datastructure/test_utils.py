import os
import h5py
import numpy as np

def create_EMD(file_name, tmpdir, chunks=None):
        h5_file = h5py.File(os.path.join(tmpdir, file_name), 'w')
        (h5_file
         .create_group('4DSTEM_experiment')
         .create_group('data')
         .create_group('datacubes')
         .create_group('datacube_0')
         .create_dataset('data', data=np.ones(shape=(2,2,3,3)), dtype='uint8', chunks=chunks)
        )
        datacube = h5_file['4DSTEM_experiment']['data']['datacubes']['datacube_0']
        datacube.attrs.create('emd_group_type', 1)
        dim1 = datacube.create_dataset('dim1', shape=(2,), dtype='uint8')
        dim1.attrs.create('name', np.string_("R_x"))
        dim1.attrs.create('units', np.string_("[pix]"))
        dim2 = datacube.create_dataset('dim2', shape=(2,), dtype='uint8')
        dim2.attrs.create('name', np.string_("R_y"))
        dim2.attrs.create('units', np.string_("[pix]"))
        dim3 = datacube.create_dataset('dim3', shape=(3,), dtype='uint8')
        dim3.attrs.create('name', np.string_("Q_x"))
        dim3.attrs.create('units', np.string_("[pix]"))
        dim4 = datacube.create_dataset('dim4', shape=(3,), dtype='uint8')
        dim4.attrs.create('name', np.string_("Q_y"))
        dim4.attrs.create('units', np.string_("[pix]"))
        return h5_file, datacube