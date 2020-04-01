# Defines the DataCube class.
#
# DataCube objects contain a 4DSTEM dataset, attributes describing its shape, and methods
# pointing to processing functions - generally defined in other files in the process directory.

import numpy as np
import hyperspy.api_nogui as hs
from hyperspy.misc.utils import DictionaryTreeBrowser
import h5py

class Metadata(object):

    def __init__(self, init=None, filepath=None, metadata_ind=None):
        """
        Instantiate a Metadata object.
        Instantiation proceeds according to the argument of init, as described below.

        Accepts:
            init            (str) controls instantiation behavior, as follows:
                            'py4DSTEM'       initialize from a py4DSTEM file (v0.4 or greater)
                            'hs'             initialize from a hyperspy file
                            'empad'          initialize from an empad file
                            None             perform no initialization; metadata dictionaries will
                                             be created, but left empty
            filepath        (str) path to the file.
                            If init is not None, filepath must point to a valid file of the
                            specified type.
                            If init is 'py4DSTEM', filepath should be an *open* h5py file object.
            metadata_ind    (int) required for init='py4DSTEM' and py4DSTEM version >= 0.4.
                            Specifies which metadata group to use.
        """
        assert(init in ['py4DSTEM','hs','empad',None])
        if init is not None:
            assert(filepath is not None)

        # Setup metadata containers
        self.setup_containers()
        self.setup_search_dicts()

        # Populate metadata according to init
        if init=='py4DSTEM':
            self.setup_metadata_py4DSTEM(filepath,metadata_ind)
        elif init=='hs':
            self.setup_metadata_hs(filepath)
        elif init=='empad':
            self.setup_metadata_empad(filepath)


    ################ Setup methods ############### 

    def setup_containers(self):
        """
        Creates the containers for metadata.
        """
        # The metadata
        self.microscope = dict()
        self.sample = dict()
        self.user = dict()
        self.calibration = dict()
        self.comments = dict()

        # Original metadata - whatever is initially found at read time
        self.original_metadata = MetadataCollection('original_metadata')

        # Search dictionaries
        self._search_dicts = MetadataCollection('search_dicts')

    def setup_search_dicts(self):
        """
        Make dictionaties which will be used for searching, scraping, and populating the active
        metadata dictionaries, using the original metadata.
        Keys here become the keys in the final metadata dictioaries; values are lists
        containing the corresponding keys to find in the hyperspy trees of the original metadata.
        These objects live in the Metadata class scope.

        Note that items that are not found will still be stored as a key in the relevant metadata
        dictionary, with the empty string as its value.  This allows these fields to populate
        in the relevant places - i.e. the metadata editor dialog. Thus any desired fields which
        will not be in the original metadata should be entered as keys with an empty seach list.
        """
        self._search_dicts.original_to_microscope_search_dict = {
            'accelerating_voltage' : [ 'beam_energy' ],
            'accelerating_voltage_units' : [ '' ],
            'camera_length' : [ 'camera_length' ],
            'camera_length_units' : [ '' ],
            'C2_aperture' : [ '' ],
            'convergence_semiangle_mrad' : [ '' ],
            'spot_size' : [ '' ],
            'scan_rotation_degrees' : [ '' ],
            'dwell_time' : [ '' ],
            'dwell_time_units' : [ '' ],
            'scan_size_Ny' : [ '' ],
            'scan_size_Nx' : [ '' ],
            'R_pix_size' : [ '' ],
            'R_pix_units' : [ '' ],
            'K_pix_size' : [ '' ],
            'K_pix_units' : [ '' ],
            'probe_FWHM_nm' : [ '' ],
            'acquisition_date' : [ '' ],
            'original_filename' : [ 'original_filename' ],
        }

        self._search_dicts.original_to_sample_search_dict = {
            'sample' : [ '' ],
            'preparation_method' : [ '' ],
            'growth_method' : [ '' ],
            'grown_by' : [ '' ],
            'other_notes' : [ '' ]
        }

        self._search_dicts.original_to_user_search_dict = {
            'name' : [ '' ],
            'institution' : [ '' ],
            'department' : [ '' ],
            'contact_email' : [ '' ],
            'contact_number' : [ '' ]
        }

        self._search_dicts.original_to_calibration_search_dict = {
            'R_pix_size' : [ '' ],
            'R_pix_units' : [ '' ],
            'K_pix_size' : [ '' ],
            'K_pix_units' : [ '' ],
            'R_to_K_rotation_degrees' : [ '' ]
        }

        self._search_dicts.original_to_comments_search_dict = {
            'comments' : [ '' ]
        }


    ################ py4DSTEM metadata ############### 

    def setup_metadata_py4DSTEM(self, filepath, metadata_ind):

        # Handle older file versions
        py4DSTEM_version = get_py4DSTEM_version(filepath)
        if py4DSTEM_version == (0,1):
            self.setup_metadata_py4DSTEM_file_v0_1(filepath)
        elif py4DSTEM_version == (0,2) or py4DSTEM_version == (0,3):
            self.setup_metadata_py4DSTEM_file_v0_2_v0_3(filepath)
        else:
            assert(isinstance(metadata_ind,(int,np.integer)))

            topgroup = get_py4DSTEM_topgroup(filepath)
            # Copy original metadata from .h5 trees to an equivalent tree structure
            self.original_metadata.shortlist = MetadataCollection('shortlist')
            self.original_metadata.all = MetadataCollection('all')

            self.populate_original_metadata_from_h5_group(filepath[topgroup + 'metadata/metadata_{}/original/shortlist'.format(metadata_ind)],self.original_metadata.shortlist)
            self.populate_original_metadata_from_h5_group(filepath[topgroup + 'metadata/metadata_{}/original/all'.format(metadata_ind)],self.original_metadata.all)

            # Copy metadata from .h5 groups to corresponding dictionaries
            self.populate_metadata_from_h5_group(filepath[topgroup + 'metadata/metadata_{}/microscope'.format(metadata_ind)],self.microscope)
            self.populate_metadata_from_h5_group(filepath[topgroup + 'metadata/metadata_{}/sample'.format(metadata_ind)],self.sample)
            self.populate_metadata_from_h5_group(filepath[topgroup + 'metadata/metadata_{}/user'.format(metadata_ind)],self.user)
            self.populate_metadata_from_h5_group(filepath[topgroup + 'metadata/metadata_{}/calibration'.format(metadata_ind)],self.calibration)
            self.populate_metadata_from_h5_group(filepath[topgroup + 'metadata/metadata_{}/comments'.format(metadata_ind)],self.comments)

    def setup_metadata_py4DSTEM_file_v0_2_v0_3(self, filepath):

        # Copy original metadata from .h5 trees to an equivalent tree structure
        self.original_metadata.shortlist = MetadataCollection('shortlist')
        self.original_metadata.all = MetadataCollection('all')

        self.populate_original_metadata_from_h5_group(filepath['4DSTEM_experiment/metadata/original/shortlist'],self.original_metadata.shortlist)
        self.populate_original_metadata_from_h5_group(filepath['4DSTEM_experiment/metadata/original/all'],self.original_metadata.all)

        # Copy metadata from .h5 groups to corresponding dictionaries
        self.populate_metadata_from_h5_group(filepath['4DSTEM_experiment/metadata/microscope'],self.data.microscope)
        self.populate_metadata_from_h5_group(filepath['4DSTEM_experiment/metadata/sample'],self.data.sample)
        self.populate_metadata_from_h5_group(filepath['4DSTEM_experiment/metadata/user'],self.data.user)
        self.populate_metadata_from_h5_group(filepath['4DSTEM_experiment/metadata/calibration'],self.data.calibration)
        self.populate_metadata_from_h5_group(filepath['4DSTEM_experiment/metadata/comments'],self.data.comments)

    def setup_metadata_py4DSTEM_file_v0_1(self, filepath):

        # Copy original metadata from .h5 trees to an equivalent tree structure
        self.original_metadata.shortlist = MetadataCollection('shortlist')
        self.original_metadata.all = MetadataCollection('all')
        self.populate_original_metadata_from_h5_group(filepath['4D-STEM_data/metadata/original/shortlist'],self.original_metadata.shortlist)
        self.populate_original_metadata_from_h5_group(filepath['4D-STEM_data/metadata/original/all'],self.original_metadata.all)

        # Copy metadata from .h5 groups to corresponding dictionaries
        self.populate_metadata_from_h5_group(filepath['4D-STEM_data/metadata/microscope'],self.data.microscope)
        self.populate_metadata_from_h5_group(filepath['4D-STEM_data/metadata/sample'],self.data.sample)
        self.populate_metadata_from_h5_group(filepath['4D-STEM_data/metadata/user'],self.data.user)
        self.populate_metadata_from_h5_group(filepath['4D-STEM_data/metadata/calibration'],self.data.calibration)
        self.populate_metadata_from_h5_group(filepath['4D-STEM_data/metadata/comments'],self.data.comments)

    def populate_original_metadata_from_h5_group(self, h5_metadata_group, target_metadata_group):
        if len(h5_metadata_group.attrs)>0:
            target_metadata_group.metadata_items = dict()
            self.populate_metadata_from_h5_group(h5_metadata_group, target_metadata_group.metadata_items)
        for subgroup_key in h5_metadata_group.keys():
            vars(target_metadata_group)[subgroup_key] = MetadataCollection(subgroup_key)
            self.populate_original_metadata_from_h5_group(h5_metadata_group[subgroup_key], vars(target_metadata_group)[subgroup_key])

    def populate_metadata_from_h5_group(self, h5_metadata_group, target_metadata_dict):
        for attr in h5_metadata_group.attrs:
            value = h5_metadata_group.attrs[attr]
            if isinstance(value,bytes):
                value = value.decode("utf-8")
            target_metadata_dict[attr] = value







    ################ Hyperspy metadata ############### 

    def setup_metadata_hs(self, filepath):

        # Get hyperspy metadata trees
        hyperspy_file = hs.load(filepath, lazy=True)
        original_metadata_shortlist = hyperspy_file.metadata
        original_metadata_all = hyperspy_file.original_metadata

        # Store original metadata
        self.original_metadata.shortlist = original_metadata_shortlist
        self.original_metadata.all = original_metadata_all

        # Search hyperspy metadata trees and use to populate metadata groups
        self.get_metadata_from_hs_tree(original_metadata_all, self._search_dicts.original_to_microscope_search_dict, self.microscope)
        self.get_metadata_from_hs_tree(original_metadata_all, self._search_dicts.original_to_sample_search_dict, self.sample)
        self.get_metadata_from_hs_tree(original_metadata_all, self._search_dicts.original_to_user_search_dict, self.user)
        self.get_metadata_from_hs_tree(original_metadata_all, self._search_dicts.original_to_calibration_search_dict, self.calibration)
        self.get_metadata_from_hs_tree(original_metadata_all, self._search_dicts.original_to_comments_search_dict, self.comments)

        self.get_metadata_from_hs_tree(original_metadata_shortlist, self._search_dicts.original_to_microscope_search_dict, self.microscope)
        self.get_metadata_from_hs_tree(original_metadata_shortlist, self._search_dicts.original_to_sample_search_dict, self.sample)
        self.get_metadata_from_hs_tree(original_metadata_shortlist, self._search_dicts.original_to_user_search_dict, self.user)
        self.get_metadata_from_hs_tree(original_metadata_shortlist, self._search_dicts.original_to_calibration_search_dict, self.calibration)
        self.get_metadata_from_hs_tree(original_metadata_shortlist, self._search_dicts.original_to_comments_search_dict, self.comments)

    @staticmethod
    def get_metadata_from_hs_tree(hs_tree, metadata_search_dict, metadata_dict):
        """
        Finds the relavant metadata in the original_metadata objects and populates the
        corresponding Metadata instance attributes.
        Accepts:
            hs_tree -   a hyperspy.misc.utils.DictionaryTreeBrowser object
            metadata_search_dict -  a dictionary with the attributes to search and the keys
                                    under which to find them
            metadata_dict - a dictionary to put the found key:value pairs into
        """
        for attr, keys in metadata_search_dict.items():
            metadata_dict[attr]=""
            for key in keys:
                found, value = Metadata.search_hs_tree(key, hs_tree)
                if found:
                    metadata_dict[attr]=value
                    break

    @staticmethod
    def search_hs_tree(key, hs_tree):
        """
        Searchers heirachically through a hyperspy.misc.utils.DictionaryBrowserTree object for
        an attribute named 'key'.
        If found, returns True, Value.
        If not found, returns False, -1.
        """
        if hs_tree is None:
            return False, -1
        else:
            for hs_key in hs_tree.keys():
                if not Metadata.istree_hs(hs_tree[hs_key]):
                    if key==hs_key:
                        return True, hs_tree[hs_key]
                else:
                    found, val = Metadata.search_hs_tree(key, hs_tree[hs_key])
                    if found:
                        return found, val
            return False, -1

    @staticmethod
    def istree_hs(node):
        if type(node)==DictionaryTreeBrowser:
            return True
        else:
            return False



    ################ Retrieve metadata ############### 

    def get_metadata_item(self, key):
        """
        Searches for key in the metadata dictionaries.  If present, returns its
        corresponding value; otherwise, returns None.

        Note that any metadata read by hyperspy into hyperspy's DictionaryTreeBrowser class
        instances is not searched; only the dictionaries living in the dictionaries
            self.microscope
            self.sample
            self.user
            self.calibration
            self.comments
        are examined. Thus if py4DSTEM did not scrape the information from a DictionaryTreeBrowser,
        and it was not entered otherwise (e.g. manually), this method will not find it.

        To search DictionaryTreeBrowsers, use the search_hs_tree() method.

        To permanently add keys for py4DSTEM to automatically scape from hyperspy files, edit the
        setup_metadata_dicts() method.
        """
        ans = []
        for md_key in self.__dict__.keys():

            # For each item in self...
            item = getattr(self, md_key)

            # If it's a dictionary, search for key, and if present return its value
            if isinstance(item, dict):
                for dict_key in item.keys():
                    if dict_key == key:
                        ans.append(item[key])

            # It it's any other object type, ignore it
            else:
                pass

        # Return
        if len(ans)==0:
            return None
        elif len(ans)==1:
            return ans[0]
        else:
            return ans

    ################ Copy metadata object ############### 

    def copy(self):
        """
        Creates and returns a copy of itself.
        """
        metadata = Metadata(init=None)
        metadata.original_metadata.shortlist = self.original_metadata.shortlist
        metadata.original_metadata.all = self.original_metadata.all

        for k,v in self.microscope.items():
            metadata.microscope[k] = v
        for k,v in self.sample.items():
            metadata.sample[k] = v
        for k,v in self.user.items():
            metadata.user[k] = v
        for k,v in self.calibration.items():
            metadata.calibration[k] = v
        for k,v in self.comments.items():
            metadata.comments[k] = v

        return metadata



########################## END OF METADATA OBJECT ########################


class MetadataCollection(object):
    """
    Empty container for storing metadata.
    """
    def __init__(self,name):
        self.__name__ = name

def get_py4DSTEM_version(h5_file):
    """
    Accepts either a filepath or an open h5py File object. Returns true if the file was written by
    py4DSTEM.
    """
    if isinstance(h5_file, h5py._hl.files.File):
        if ('version_major' in h5_file.attrs) and('version_minor' in h5_file.attrs):
            #implicitlty checks if v0.5 or above
            version_major = h5_file.attrs['version_major']
            version_minor = h5_file.attrs['version_minor']
        else:
            #if not in root directory, will be in one of top groups
            for key in h5_file.keys():
                if ('version_major' in h5_file[key].attrs) and ('version_minor' in h5_file[key].attrs) and ('emd_group_type' in h5_file[key].attrs):
                    if h5_file[key].attrs['emd_group_type'] == 2:
                        version_major = h5_file[key].attrs['version_major']
                        version_minor = h5_file[key].attrs['version_minor']
                        break
        return version_major, version_minor
    else:
        try:
            f = h5py.File(h5_file, 'r')
            result = get_py4DSTEM_version(f)
            f.close()
            return result
        except OSError:
            print("Error: file cannot be opened with h5py, and may not be in HDF5 format.")
            return (0,0)

##TODO: take this out or make it consistent/generalized
def get_py4DSTEM_topgroup(h5_file):
    """
    Accepts an open h5py File boject. Returns string of the top group name. 
    """
    if ('4DSTEM_experiment' in h5_file.keys()): # or ('4D-STEM_data' in h5_file.keys()) or ('4DSTEM_simulation' in h5_file.keys())):
        return '4DSTEM_experiment/'
    elif ('4DSTEM_simulation' in h5_file.keys()):
        return '4DSTEM_simulation/'
    else:
        return '4D-STEM_data/'


