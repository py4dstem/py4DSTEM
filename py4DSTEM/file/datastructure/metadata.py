# Defines the DataCube class.
#
# DataCube objects contain a 4DSTEM dataset, attributes describing its shape, and methods
# pointing to processing functions - generally defined in other files in the process directory.

import numpy as np
from hyperspy.misc.utils import DictionaryTreeBrowser
from .dataobject import DataObject

class Metadata(DataObject):

    def __init__(self, is_py4DSTEM_file,
                 original_metadata_shortlist=None, original_metadata_all=None,
                 filepath=None):
        """
        Instantiate a Metadata object.
        Metadata is populated in one of two ways - either for native py4DSTEM files or for
        non-native files - and the kwargs this method expects depend on this.

        Accepts (all):
            is_py4DSTEM_file        (bool) flag indicating native or non-native py4DSTEM files

        Accepts (non-py4DSTEM files):
            original_metadata_shortlist     (hyperspy Signal.metadata tree)
            original_metadata_all           (hyperspy Signal.original_metadata tree)

        Accepts (py4DSTEM file):
            filepath                 (str) path to the py4DSTEM h5 file
        """
        DataObject.__init__(self)
        self.metadata = self

        # Setup metadata containers
        self.setup_metadata_containers()
        self.setup_metadata_search_dicts()

        # For non-py4DSTEM files
        if not is_py4DSTEM_file:
            self.setup_metadata_hs_file(original_metadata_shortlist, original_metadata_all)

        # For py4DSTEM files
        else:
            py4DSTEM_version = get_py4DSTEM_version(filepath)
            if py4DSTEM_version == (0,1):
                self.setup_metadata_py4DSTEM_file_v0_1(filepath)
            elif py4DSTEM_version == (0,2) or py4DSTEM_version == (0,3):
                self.setup_metadata_py4DSTEM_file(filepath)
            else:
                raise ValueError("Unrecognized py4DSTEM version, {}.{}".format(py4DSTEM_version[0],
                                                                           py4DSTEM_version[1]))

    ################ Setup methods ############### 

    def setup_metadata_containers(self):
        """
        Creates the containers for metadata.
        """
        self.data = MetadataCollection('data')
        self.original_metadata = MetadataCollection('original metadata')
        self.data.microscope = dict()
        self.data.sample = dict()
        self.data.user = dict()
        self.data.calibration = dict()
        self.data.comments = dict()

    def setup_metadata_search_dicts(self):
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
        self.original_to_microscope_search_dict = {
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

        self.original_to_sample_search_dict = {
            'sample' : [ '' ],
            'preparation_method' : [ '' ],
            'growth_method' : [ '' ],
            'grown_by' : [ '' ],
            'other_notes' : [ '' ]
        }

        self.original_to_user_search_dict = {
            'name' : [ '' ],
            'institution' : [ '' ],
            'department' : [ '' ],
            'contact_email' : [ '' ],
            'contact_number' : [ '' ]
        }

        self.original_to_calibration_search_dict = {
            'R_pix_size' : [ '' ],
            'R_pix_units' : [ '' ],
            'K_pix_size' : [ '' ],
            'K_pix_units' : [ '' ],
            'R_to_K_rotation_degrees' : [ '' ]
        }

        self.original_to_comments_search_dict = {
            'comments' : [ '' ]
        }


    def setup_metadata_py4DSTEM_file(self, filepath):

        # Copy original metadata from .h5 trees to an equivalent tree structure
        self.original_metadata.shortlist = MetadataCollection('shortlist')
        self.original_metadata.all = MetadataCollection('all')

        self.get_original_metadata_from_filepath(filepath['4DSTEM_experiment']['metadata']['original']['shortlist'],self.original_metadata.shortlist)
        self.get_original_metadata_from_filepath(filepath['4DSTEM_experiment']['metadata']['original']['all'],self.original_metadata.all)

        # Copy metadata from .h5 groups to corresponding dictionaries
        self.get_metadata_from_filepath(filepath['4DSTEM_experiment']['metadata']['microscope'],self.data.microscope)
        self.get_metadata_from_filepath(filepath['4DSTEM_experiment']['metadata']['sample'],self.data.sample)
        self.get_metadata_from_filepath(filepath['4DSTEM_experiment']['metadata']['user'],self.data.user)
        self.get_metadata_from_filepath(filepath['4DSTEM_experiment']['metadata']['calibration'],self.data.calibration)
        self.get_metadata_from_filepath(filepath['4DSTEM_experiment']['metadata']['comments'],self.data.comments)

    def setup_metadata_py4DSTEM_file_v0_1(self, filepath):

        # Copy original metadata from .h5 trees to an equivalent tree structure
        self.original_metadata.shortlist = MetadataCollection('shortlist')
        self.original_metadata.all = MetadataCollection('all')
        self.get_original_metadata_from_filepath(filepath['4D-STEM_data']['metadata']['original']['shortlist'],self.original_metadata.shortlist)
        self.get_original_metadata_from_filepath(filepath['4D-STEM_data']['metadata']['original']['all'],self.original_metadata.all)

        # Copy metadata from .h5 groups to corresponding dictionaries
        self.get_metadata_from_filepath(filepath['4D-STEM_data']['metadata']['microscope'],self.data.microscope)
        self.get_metadata_from_filepath(filepath['4D-STEM_data']['metadata']['sample'],self.data.sample)
        self.get_metadata_from_filepath(filepath['4D-STEM_data']['metadata']['user'],self.data.user)
        self.get_metadata_from_filepath(filepath['4D-STEM_data']['metadata']['calibration'],self.data.calibration)
        self.get_metadata_from_filepath(filepath['4D-STEM_data']['metadata']['comments'],self.data.comments)

    def setup_metadata_hs_file(self, original_metadata_shortlist=None, original_metadata_all=None):

        # Store original metadata
        self.original_metadata.shortlist = original_metadata_shortlist
        self.original_metadata.all = original_metadata_all

        # Search original metadata and use to populate metadata groups
        self.get_metadata_from_original_metadata(original_metadata_all, self.original_to_microscope_search_dict, self.data.microscope)
        self.get_metadata_from_original_metadata(original_metadata_all, self.original_to_sample_search_dict, self.data.sample)
        self.get_metadata_from_original_metadata(original_metadata_all, self.original_to_user_search_dict, self.data.user)
        self.get_metadata_from_original_metadata(original_metadata_all, self.original_to_calibration_search_dict, self.data.calibration)
        self.get_metadata_from_original_metadata(original_metadata_all, self.original_to_comments_search_dict, self.data.comments)

        self.get_metadata_from_original_metadata(original_metadata_shortlist, self.original_to_microscope_search_dict, self.data.microscope)
        self.get_metadata_from_original_metadata(original_metadata_shortlist, self.original_to_sample_search_dict, self.data.sample)
        self.get_metadata_from_original_metadata(original_metadata_shortlist, self.original_to_user_search_dict, self.data.user)
        self.get_metadata_from_original_metadata(original_metadata_shortlist, self.original_to_calibration_search_dict, self.data.calibration)
        self.get_metadata_from_original_metadata(original_metadata_shortlist, self.original_to_comments_search_dict, self.data.comments)


    ################ Populate metadata ############### 

    def get_original_metadata_from_filepath(self, h5_metadata_group, datacube_metadata_group):
        if len(h5_metadata_group.attrs)>0:
            datacube_metadata_group.metadata_items = dict()
            self.get_metadata_from_filepath(h5_metadata_group, datacube_metadata_group.metadata_items)
        for subgroup_key in h5_metadata_group.keys():
            vars(datacube_metadata_group)[subgroup_key] = MetadataCollection(subgroup_key)
            self.get_original_metadata_from_filepath(h5_metadata_group[subgroup_key], vars(datacube_metadata_group)[subgroup_key])


    def get_metadata_from_filepath(self, h5_metadata_group, datacube_metadata_dict):
        for attr in h5_metadata_group.attrs:
            datacube_metadata_dict[attr] = h5_metadata_group.attrs[attr]

    @staticmethod
    def get_metadata_from_original_metadata(hs_tree, metadata_search_dict, metadata_dict):
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

    @staticmethod
    def add_metadata_item(key,value,metadata_dict):
        """
        Adds a single item, given by the pair key:value, to the metadata dictionary metadata_dict
        """
        metadata_dict[key] = value


    ################ Retrieve metadata ############### 

    def get_metadata_item(self, key):
        """
        Searches for key in the dictionaries found in self.data.  If present, returns its
        corresponding value; otherwise, returns None.

        Note that any metadata read by hyperspy into hyperspy's DictionaryTreeBrowser class
        instances is not searched; only the dictionaries living in self.data are examined. Thus
        if py4DSTEM did not scrape the information from a DictionaryTreeBrowser, and it was not
        entere otherwise (e.g. manually), this method will not find it.

        To search DictionaryTreeBrowsers, use the search_hs_tree() method.

        To permanently add keys for py4DSTEM to automatically scape from hyperspy files, edit the
        setup_metadata_dicts() method.
        """
        ans = []
        for md_key in self.data.__dict__.keys():

            # For each item in self.metadat...
            item = getattr(self.data, md_key)

            # If it's a dictionary, search for key, and if presnt return its value
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


########################## END OF METADATA OBJECT ########################


class MetadataCollection(object):
    """
    Empty container for storing metadata.
    """
    def __init__(self,name):
        self.__name__ = name

def get_py4DSTEM_version(filepath):
    version_major = filepath.attrs['version_major']
    version_minor = filepath.attrs['version_minor']
    return version_major, version_minor


