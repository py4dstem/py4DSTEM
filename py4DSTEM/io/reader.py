# Reads 4D-STEM data with hyperspy, stores as a Datacube object

import h5py
import numpy as np
import hyperspy
import hyperspy.api as hs
from os.path import splitext
from process.datastructure.datacube import Datacube

def read_data(filename):
    """
    Takes a filename as input, and outputs a DataCube object.

    If filename is a .h5 file,
    read_data() checks if the file was written by py4DSTEM.  If it was, the metadata are read
    and saved directly.  Otherwise, the file is read with hyperspy, and metadata is scraped and
    saved from the hyperspy file.
    """
    if splitext(filename)[1] == ".h5":
        print("{} is an HDF5 file.  Reading...".format(filename))
        return read_py4DSTEM_file(filename)
    # Load data
    else:
        print("{} is not an HDF5 file.  Reading with hyperspy...".format(filename))
        return read_other_file(filename)

# TODO: add this! #
def read_py4DSTEM_file(filename):
    print("Reading h5 files is not yet supported! Check back very soon.")
    pass

def read_other_file(filename):
    try:
        hyperspy_file = hs.load(filename)
        if len(hyperspy_file.data.shape)==3:
            R_N, Q_Ny, Q_Nx = hyperspy_file.data.shape
            R_Ny, R_Nx = R_N, 1
        elif len(hyperspy_file.data.shape)==4:
            R_Ny, R_Nx, Q_Ny, Q_Nx = hyperspy_file.data.shape
        else:
            print("Error: unexpected raw data shape of {}".format(hyperspy_file.data.shape))
            return Datacube(data=np.random.rand(100,512,512),10,10,512,512, filename=filename)
        return Datacube(data=hyperspy_file.data, R_Ny, R_Nx, Q_Ny, Q_Nx,
                            original_metadata_shortlist=hyperspy_file.metadata,
                            original_metadata_all=hyperspy_file.original_metadata,
                            filename=filename)
    except Exception as err:
        print("Failed to load", err)
        print("Initializing random datacube.")
        return Datacube(data=np.random.rand(100,512,512),10,10,512,512)






class h5writer(object):

    def __init__(self,inputfile,outputfile=None):

        # Read input and set up new .h5 file
        self.read_file(inputfile)
        self.make_h5_file(outputfile)

        # Create group structure, add data and accessible metadata
        self.write_datacube_group()
        self.write_metadata_group()
        self.write_processing_group()

        # Enter UI loop - gets and store additional metadata from user input
        #self.UI_loop()

        # Basic pre-processing: reshape, calibrate, bin, crop 
        # These methods run iff the needed metadata is present
        #self.reshape()
        self.calibrate()
        #self.bin()
        #self.crop()

        # Close file
        print("Done.")
        self.f.close()

    def read_file(self,inputfile):
        print("Reading file {}...".format(inputfile))
        self.inputfile = inputfile
        self.filepath, self.fileext = splitext(inputfile)
        self.hyperspy_file = hs.load(inputfile)

    def make_h5_file(self,outputfile):
        if outputfile is not None:
            self.outputfile = outputfile
        else:
            self.outputfile = self.filepath+".h5"
        print("Creating file {}...".format(self.outputfile))
        self.f = h5py.File(self.outputfile,"w")
        self.f.attrs.create("version_major",0)
        self.f.attrs.create("version_minor",2)
        self.group_data = self.f.create_group("4D-STEM_data")

    def write_datacube_group(self):

        self.group_datacube = self.group_data.create_group("datacube")
        self.group_datacube.attrs.create("emd_group_type",1)

        self.data_datacube = self.group_datacube.create_dataset("datacube", data=self.hyperspy_file.data)
        # TODO: consider defining data chunking here, keeping k-space slices together

        # Dimensions
        if len(self.data_datacube.shape)==4:
            self.R_Ny,self.R_Nx,self.K_Ny,self.K_Nx = self.data_datacube.shape
        elif len(self.data_datacube.shape)==3:
            self.R_Nx,self.K_Ny,self.K_Nx = self.data_datacube.shape
            self.R_Ny = 1
        else:
            print("Error: datacube has {} dimensions.".format(len(self.data_datacube.shape)))
        self.data_R_Ny = self.group_datacube.create_dataset("dim1",(self.R_Ny,))
        self.data_R_Nx = self.group_datacube.create_dataset("dim2",(self.R_Nx,))
        self.data_K_Ny = self.group_datacube.create_dataset("dim3",(self.K_Ny,))
        self.data_K_Nx = self.group_datacube.create_dataset("dim4",(self.K_Nx,))

        # Populate uncalibrated dimensional axes
        self.data_R_Ny[...] = np.arange(0,self.R_Ny)
        self.data_R_Ny.attrs.create("name",np.string_("R_y"))
        self.data_R_Ny.attrs.create("units",np.string_("[pix]"))
        self.data_R_Nx[...] = np.arange(0,self.R_Nx)
        self.data_R_Nx.attrs.create("name",np.string_("R_x"))
        self.data_R_Nx.attrs.create("units",np.string_("[pix]"))
        self.data_K_Ny[...] = np.arange(0,self.K_Ny)
        self.data_K_Ny.attrs.create("name",np.string_("K_y"))
        self.data_K_Ny.attrs.create("units",np.string_("[pix]"))
        self.data_K_Nx[...] = np.arange(0,self.K_Nx)
        self.data_K_Nx.attrs.create("name",np.string_("K_x"))
        self.data_K_Nx.attrs.create("units",np.string_("[pix]"))

    def write_metadata_group(self):
        print("Writing metadata...")
        self.group_metadata = self.group_data.create_group("metadata")
        self.group_original_metadata = self.group_metadata.create_group("original")
        self.group_microscope_metadata = self.group_metadata.create_group("microscope")
        self.group_sample_metadata = self.group_metadata.create_group("sample")
        self.group_user_metadata = self.group_metadata.create_group("user")
        self.group_processing_metadata = self.group_metadata.create_group("processing")
        self.group_calibration_metadata = self.group_metadata.create_group("calibration")
        self.group_comments_metadata = self.group_metadata.create_group("comments")

        self.group_original_metadata_all = self.group_original_metadata.create_group("all")
        self.group_original_metadata_shortlist = self.group_original_metadata.create_group("shortlist")

        self.transfer_metadata(self.hyperspy_file.metadata,self.group_original_metadata_shortlist)
        self.transfer_metadata(self.hyperspy_file.original_metadata,self.group_original_metadata_all)

        self.populate_microscope_metadata()
        #self.populate_sample_metadata()
        #self.populate_user_metadata()
        #self.populate_processing_metadata()
        #self.populate_comments_metadata()

    def transfer_metadata(self,tree,group):
        """
        Transfers metadata from any data format readable in hyperspy into a well-formated
        tree of metadata in the new .h5 file.

        Accepts two arguments:
            tree - a dictionary tree of metadata, in the format output by hyperspy
            group - an hdf5 file group to place metadata in
        The information in metadata_tree is then added to group, preserving the tree's
        structure, such that non-terminal nodes become groups, and terminal nodes become
        attributes.
        """
        for key in tree.keys():
            if self.istree_hs(tree[key]):
                subgroup = group.create_group(key)
                self.transfer_metadata(tree[key],subgroup)
            else:
                if type(tree[key])==str:
                    group.attrs.create(key,np.string_(tree[key]))
                else:
                    group.attrs.create(key,tree[key])

    @staticmethod
    def istree_hs(node):
        """
        Determines if a node in a hyperspy metadata structure is a parent or terminal leaf.
        """
        if type(node)==hyperspy.misc.utils.DictionaryTreeBrowser:
            return True
        else:
            return False

    def populate_microscope_metadata(self):
        """
        Populates the microscope metadata group by searching the original metadata group (which
        has been populated by the transfer_metadata() method) for known keywords.

        Uses the dictionary self.microscope_metadata_dict to perform search; for new filetypes
        with different keys, edit the make_microscope_metadata_dict() method.
        """
        self.make_microscope_metadata_dict()

        # Uncomment to create all possible metadata fields, initialized to -1
        #for attr in self.microscope_metadata_dict.keys():
        #    self.group_microscope_metadata.attrs.create(attr,-1)

        for attr, keys in self.microscope_metadata_dict.items():
            for key in keys:
                found, value = self.search_metadata_tree(key,self.group_original_metadata_shortlist)
                if found:
                    self.group_microscope_metadata.attrs.create(attr,value)
                    break

    def search_metadata_tree(self,key,group):
        """
        Searches hierarchically through "group" for an attribute named "key".
        If found, returns True, Value.
        If not found, returns False, -1.
        """
        if type(group)==h5py._hl.dataset.Dataset:
            pass
        else:
            if key in list(group.attrs.keys()):
                return True, group.attrs[key]
            else:
                for groupkey in group.keys():
                    found,val = self.search_metadata_tree(key,group[groupkey])
                    if found:
                        return found,val
        return False, -1


    def transfer_metadata(self,tree,group):
        """
        Transfers metadata from any data format readable in hyperspy into a well-formated
        tree of metadata in the new .h5 file.

        Accepts two arguments:
            tree - a dictionary tree of metadata, in the format output by hyperspy
            group - an hdf5 file group to place metadata in
        The information in metadata_tree is then added to group, preserving the tree's
        structure, such that non-terminal nodes become groups, and terminal nodes become
        attributes.
        """
        for key in tree.keys():
            if self.istree_hs(tree[key]):
                subgroup = group.create_group(key)
                self.transfer_metadata(tree[key],subgroup)
            else:
                if type(tree[key])==str:
                    group.attrs.create(key,np.string_(tree[key]))
                else:
                    group.attrs.create(key,tree[key])

    @staticmethod
    def istree_hs(node):
        if type(node)==hyperspy.misc.utils.DictionaryTreeBrowser:
            return True
        else:
            return False


    def make_microscope_metadata_dict(self):
        """
        Make the dictionary used to look up metadata from the original metadata files.
        The keys correspond to the attributes that will be defined in the .h5 file.
        The values are lists containing the corresponding keys in different supported filetypes.
        When a new type of file is being used, the keys it uses for each metadata item should be
        added to the correct list.
        """
        self.microscope_metadata_dict = {
            'accelerating_voltage_kV' : [ 'beam_energy' ],
            'camera_length_mm' : [ 'camera_length' ],
            'C2_aperture' : [ '' ],
            'convergence_semiangle_mrad' : [ '' ],
            'spot_size' : [ '' ],
            'scan_rotation_degrees' : [ '' ],
            'dwell_time_ms' : [ '' ],
            'scan_size_Ny' : [ '' ],
            'scan_size_Nx' : [ '' ],
            'R_pix_size' : [ '' ],
            'R_units' : [ '' ],
            'K_pix_size' : [ '' ],
            'K_units' : [ '' ],
            'probe_FWHM_nm' : [ '' ]
        }

    def write_processing_group(self):
        self.group_processing = self.group_data.create_group("processing")
        self.group_k_objects = self.group_processing.create_group("k-space_objects")
        self.group_r_objects = self.group_processing.create_group("r-space_objects")

    def write_calibration_metadata(self):
        self.group_microscope_metadata.attrs.create("R_pix_size",0.2)
        self.group_microscope_metadata.attrs.create("R_units",np.string_('nm'))
        self.group_microscope_metadata.attrs.create("K_pix_size",3)

    def calibrate(self):
        # Calibrate R axes
        found_R_pixsize, R_pixsize = self.search_metadata_tree("R_pix_size", self.group_microscope_metadata)
        if found_R_pixsize:
            self.data_R_Ny[...] = np.arange(0,self.R_Ny*R_pixsize,R_pixsize)
            self.data_R_Nx[...] = np.arange(0,self.R_Nx*R_pixsize,R_pixsize)
        # Set R axis units
        found_R_units, R_units = self.search_metadata_tree("R_units", self.group_microscope_metadata)
        if found_R_units:
            self.data_R_Nx.attrs["units"] = R_units
            self.data_R_Ny.attrs["units"] = R_units
        # Calibrate K axes
        found_K_pixsize, K_pixsize = self.search_metadata_tree("K_pix_size", self.group_microscope_metadata)
        if found_K_pixsize:
            self.data_K_Ny[...] = np.arange(0,self.K_Ny*K_pixsize,K_pixsize)
            self.data_K_Nx[...] = np.arange(0,self.K_Nx*K_pixsize,K_pixsize)
        # Set K axis units
        found_K_units, K_units = self.search_metadata_tree("K_units", self.group_microscope_metadata)
        if found_K_units:
            self.data_K_Ny.attrs["units"] = K_units
            self.data_K_Nx.attrs["units"] = K_units





if __name__=="__main__":
    writer = h5writer('test.dm3')






######## Print file structure ########

#f.close()



############################### File structure ###############################
#
# /
# |--attr: version_major=0
# |--attr: version_minor=2
# |--grp: 4D-STEM_data
#             |
#             |--grp: datacube
#             |          |--attr: emd_group_type=1
#             |          |--data: datacube
#             |          |--data: dim1
#             |          |    |--attr: name="R_y"
#             |          |    |--attr: units="[n_m]"
#             |          |--data: dim2
#             |          |    |--attr: name="R_y"
#             |          |    |--attr: units="[n_m]"
#             |          |--data: dim3
#             |          |    |--attr: name="R_y"
#             |          |    |--attr: units="[n_m]"
#             |          |--data: dim4
#             |               |--attr: name="R_y"
#             |               |--attr: units="[n_m]"
#             |
#             |--grp: processing
#             |          |--# This will contain objects created and used during processing
#             |          |--# e.g. vacuum probe, shifts from scan coils, binary masks,
#             |          |--#      convolution kernels, etc.
#             |
#             |--grp: metadata
#                        |--grp: original
#                        |   |--# Raw metadata from original files
#                        |
#                        |--grp: microscope
#                        |   |--# Acquisition parameters
#                        |   |--# Accelerating voltage, camera length, convergence angle, 
#                        |   |--# C2 aperture, spot size, exposure time, scan rotation angle,
#                        |   |--# scan shape, probe FWHM
#                        |
#                        |--grp: sample
#                        |   |--# Material, preparation
#                        |
#                        |--grp: user
#                        |   |--# Name, instituion, dept, contact email
#                        |
#                        |--grp: processing
#                        |   |--# original file name, processing perfomed - binning, cropping
#                        |   |--# Consider attaching logs
#                        |
#                        |--grp: calibration
#                        |   |--# R pixel size, K pixel size, R/K rotation offset
#                        |   |--# In case of duplicates here and in grp: microscope (e.g. pixel
#                        |   |--# sizes), quantities here are calculated from data rather than
#                        |   |--# being read from the instrument
#                        |
#                        |--grp: comments


