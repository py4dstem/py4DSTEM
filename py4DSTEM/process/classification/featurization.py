import numpy as np
from py4DSTEM.io.datastructure import DataCube
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.decomposition import NMF, PCA, FastICA
from sklearn.mixture import GaussianMixture
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from skimage.filters import threshold_otsu, threshold_yen
from skimage.measure import label
from skimage.morphology import closing, square, remove_small_objects

class Featurization(object):
    """
    A class for feature selection, modification, and classification of 4D-STEM data based on a user defined
    set of input features for each pattern. Features are stored under Featurization.features and can be
    accessed with their key assigned upon initialization. 
    
    Initialization methods:
        __init__:
            Creates dictionaries to store features in "self.features".
    
    Feature Dictionary Modification Methods
        add_feature:
            Adds a matrix to be stored at self.features[key]
        remove_feature:
            Removes a key-value pair at the self.features[key]
        update_features:
            Updates an NMF, PCA, or ICA reduced feature set to the self.features location.    
    
    Feature Preprocessing Methods
        MinMaxScaler:
            Performs sklearn MinMaxScaler operation on features stored at a key
        RobustScaler:
            Performs sklearn RobustScaler operation on features stored at a key
        mean_feature:
            Takes the rowwise average of a matrix stored at a key, such that only one column is left,
            reducing a set of n features down to 1 feature per pattern.
        median_feature:
            Takes the rowwise median of a matrix stored at a key, such that only one column is left,
            reducing a set of n features down to 1 feature per pattern.
        max_feature:
            Takes the rowwise max of a matrix stored at a key, such that only one column is left,
            reducing a set of n features down to 1 feature per pattern.
        
    Classification Methods
        PCA:
            Principal Component Analysis to refine features.
        ICA:
            Independent Component Analysis to refine features.
        NMF:
            (traditional) Nonnegative Matrix Factorization to refine features.
        nmf_iterative:
            Iterative Nonnegative Matrix Factorization to refine features. Performed iteratively by merging
            [add more details later]
        gmm:
            Gaussian mixture model to predict class labels. Fits a gaussian based on covariance of features
            [add more details later]
    
    Class Examination Methods
        get_class_DPs:
            Gets weighted class diffraction patterns (DPs) for an NMF or GMM operation
        get_class_ims:
            Gets weighted class images (ims) for an NMF or FMM operation
    """
    
    def __init__(self, features, R_Nx, R_Ny):
        """
        Initializes classification instance.
        
        This method:
        1. Generates key:value pair to access input features
        2. Initializes the empty dictionaries for feature modification and classification
        
        Args:
            features (list): A list of ndarrays which will each be associated with value stored at the key in the same index within the list
            R_Nx (int): The real space x dimension of the dataset
            R_Ny (int): The real space y dimension of the dataset
        """
        self.R_Nx = R_Nx
        self.R_Ny = R_Ny

        if isinstance(features, np.ndarray):
            if len(features.shape) == 3:
                self.features = features.reshape(R_Nx*R_Ny, features.shape[-1])
            elif len(features.shape) == 2:
                self.features = features
            else: 
                raise ValueError(
                        'feature array must be of dimensions (R_Nx*R_Ny, num_features) or (R_Nx, R_Ny, num_features)'
                        )  
        elif isinstance(features, list):
            #check types - all must be either numpy or featurization instances?
            for i in range(len(features)):
                if features[i].shape == 3:
                    features[i] = features[i].reshape(R_Nx*R_Ny, features.shape[-1])
                if features[i].shape != 2:
                    raise ValueError(
                        'feature array(s) in list must be of dimensions (R_Nx*R_Ny, num_features) or (R_Nx, R_Ny, num_features)'
                        )  
            self.features = np.concatenate(features, axis=1)
        else:
            raise TypeError('features must be either a single np.ndarray of shape 2 or 3 or a list of np.ndarrays')
        return

    def add_features(self, feature):
        """
        Add a feature to the end of the features array
        
        Args:
            key (int, float, str): A key in which a feature can be accessed from
            feature (ndarray): The feature associated with the key
        """
        self.features = np.concatenate(self.features, feature, axis = 1)
        return
    
    def delete_features(self, index):
        """
        Deletes feature columns from the feature array
        
        Args:
            index (int, list): A key which will be removed
        """
        self.features = np.delete(self.features, index, axis = 1)
        return
    
    # User code:
    # new_feature = Featurization.concatenate_features(
    #    [f1, f2, f3],
    #    name = 'meow'
    # )

    # def concatenate_features(self, list, output_key):
    #     """
    #     Concatenates dataframes in 'key' and saves them to features with key 'output_key'
        
    #     Args:
    #         keys (list) A list of keys to be concatenated into one array
    #         output_key (int, float, str) the key in which the concatenated array will be stored
    #     """
    #     self.features[output_key] = np.concatenate([self.features[keys[i]] for i in range(len(keys))], axis = 1)
    #     return
    
    # def mean_feature(self, keys, replace = False):
    #     """
    #     Takes columnwise mean of feature in key. if replace = True, replaces value in key with
    #     mean value over entire feature
        
    #     Args:
    #         keys (list of str): List of strings in which to conduct the calculation on
    #         replace (bool): Whether or not to replace the feature in place or create a new feature set
    #     """
    #     if replace == True:
    #         for i in range(len(keys)):
    #             self.features[keys[i]] = np.mean(self.features[keys[i]], axis = 1).reshape(self.features[keys[i]].shape[0],1)
    #     elif replace == False:
    #         for i in range(len(keys)):
    #             self.features[keys[i] + '_mean'] = np.mean(self.features[keys[i]], axis = 1).reshape(self.features[keys[i]].shape[0],1)
    #     return

    # def median_feature(self, keys, replace = False):
    #     """
    #     Takes columnwise median of feature in key. if replace = True, replaces value in key with
    #     median value over entire feature
        
    #     Args:
    #         keys (list of str): List of strings in which to conduct the calculation on
    #         replace (bool): Whether or not to replace the feature in place or create a new feature set
    #     """
    #     if replace == True:
    #         for i in range(len(keys)):
    #             self.features[keys[i]] = np.median(self.features[keys[i]], axis = 1).reshape(self.features[keys[i]].shape[0],1)
    #     elif replace == False:
    #         for i in range(len(keys)):
    #             self.features[keys[i] + '_median'] = np.median(self.features[keys[i]], axis = 1).reshape(self.features[keys[i]].shape[0],1)
    #     return
    
    # def max_feature(self, keys, replace = False):
    #     """
    #     Takes the columnwise max of the ndarray located 
        
    #     Args:
    #         keys (list of str): List of strings in which to conduct the calculation on
    #         replace (bool): Whether or not to replace the feature in place or create a new feature set
    #     """
    #     if replace == True:
    #         for i in range(len(keys)):
    #             self.features[keys[i]] = np.max(self.features[keys[i]], axis = 1).reshape(self.features[keys[i]].shape[0],1)
    #     elif replace == False:
    #         for i in range(len(keys)):
    #             self.features[keys[i] + '_max'] = np.max(self.features[keys[i]], axis = 1).reshape(self.features[keys[i]].shape[0],1)
    #     return

    def MinMaxScaler(self, return_scaled = True):
        """
        Uses sklearn MinMaxScaler to scale a subset of the input features. 
        Replaces a feature with the positive shifted array.
        
        Args:
            return_scaled (bool): returns the scaled array
        """
        mms = MinMaxScaler()
        self.features = mms.fit_transform(self.features)
        if return_scaled == True:
            return self.features
        else:
            return
    
    def RobustScaler(self, return_scaled = True):
        """
        Uses sklearn RobustScaler to scale a subset of the input features. 
        Replaces a feature with the positive shifted array.
        
        Args:
            return_scaled (bool): returns the scaled array
        """
        rs = RobustScaler()
        self.features = rs.fit_transform(self.features)
        if return_scaled == True:
            return self.features
        else:
            return
    
    def shift_positive(self, return_scaled = True):
        """
        Replaces a feature with the positive shifted array.
        
        Args:
            return_scaled (bool): returns the scaled array
        """
        self.features += np.abs(self.features.min())
        if return_scaled == True:
            return self.features
        else:
            return
    
    def PCA(self, components, return_results = True):
        """
        Performs PCA on features
        
        Args:
            components (list): A list of ints for each key. This will be the output number of features
        """
        pca = PCA(n_components = components)
        self.pca = pca.fit_transform(self.features)
        return self.pca

    def ICA(self, components):
        """
        Performs ICA on features
        
        Args:
            components (list): A list of ints for each key. This will be the output number of features
        """
        ica = FastICA(n_components = components)
        self.ica = ica.fit_transform(self.features)
        return
    
    def NMF(
        self,
        max_components, 
        merge_thresh, num_models, 
        max_iterations = 1, 
        random_state = None, 
        save_all_models = True,
        return_results = False
    ):
        """
        Performs nmf iteratively on input features
        
        Args:
            max_components (int): number of initial components to start the first NMF iteration
            merge_thresh (float): correlation threshold to merge features
            num_models (int): Number of independent models to run (number of learners that will be combined in consensus)
            max_iterations (int): number of iterations. Default 1, which runs traditional NMF
            random_state (int): random state
            save_all_models (bool): Whether or not to return all of the models - default is to return
                all outputs for consensus clustering
            return_results (bool): Whether or not to return the final class weights
        """
        self.W = _nmf_single(
            self.features,
            max_components=max_components,
            merge_thresh = merge_thresh,
            num_models = num_models,
            max_iterations = max_iterations, 
            random_state = random_state,
            save_all_models = save_all_models
            )
        if return_results == True:
            return self.W
        else:
            return
        
    def GMM(self, cv, components, num_models, random_state = None):
        """
        Performs gaussian mixture model on input features
        
        Args:
            cv (str): covariance type - must be 'spherical', 'tied', 'diag', or 'full'
            components (int): number of components
            num_models (int): number of models to run
            random_state (int): random state
        """
        self.gmm, self.gmm_labels, self.gmm_proba = _gmm_single(
            self.features,
            cv=cv,
            components = components,
            num_models = num_models, 
            random_state = random_state)
        return

    def get_class_DPs(self, datacube, classification_method, thresh):
        """
        Returns weighted class patterns based on classification instance
        datacube must be vectorized in real space (shape = (R_Nx * R_Ny, 1, Q_Nx, Q_Ny)
        
        Args:
            classification_method (str): either 'nmf' or 'gmm' - finds location of clusters
            datacube: Vectorized in real space, with shape (R_Nx * R_Ny, Q_Nx, Q_Ny)
        """
        class_patterns = []
        if classification_method== 'nmf':
            for l in range(self.W.shape[1]):
                class_pattern = np.zeros((datacube.data.shape[2], datacube.data.shape[3]))
                x_ = np.where(self.W[:,l] > thresh)[0]
                for x in range(x_.shape[0]):
                    class_pattern += datacube.data[x_[x],0] * self.W[x_[x],l]
                class_patterns.append(class_pattern  / np.sum(self.W[x_, l]))
        elif classification_method == 'gmm':
            for l in range(np.max(self.gmm_labels)):
                class_pattern = np.zeros((datacube.data.shape[2], datacube.data.shape[3]))
                x_ = np.where(self.gmm_proba[:,l] > thresh)[0]
                for x in range(x_.shape[0]):
                    class_pattern += datacube.data[x_[x],0] * self.gmm_proba[x_[x],l]
                class_patterns.append(class_pattern / np.sum(self.gmm_proba[x_,l]))
        self.class_DPs = class_patterns
        return
        
    def get_class_ims(self, classification_method):
        """
        Returns weighted class maps based on classification instance
        
        Args:
            classification_method (str): either 'nmf' or 'gmm' - finds location of clusters
        """
        class_maps = []
        if classification_method == 'nmf':
            if type(self.W) == list:
                for l in range(len(self.W)):
                    small_class_maps = []
                    for k in range(self.W[l].shape[1]):
                        small_class_maps.append(self.W[l][:,k].reshape(self.R_Nx, self.R_Ny))
                    class_maps.append(small_class_maps)
            else:
                for l in range(self.W.shape[1]):
                    class_maps.append(self.W[:,l].reshape(self.R_Nx,self.R_Ny))
        elif classification_method == 'gmm':
            if type(self.W) == list:
                raise ValueError('Method not implemented yet, sorry!')
            else:
                for l in range((np.max(self.gmm_labels))):
                    R_vals = np.where(self.gmm_labels.reshape(self.R_Nx,self.R_Ny) == l, 1, 0)
                    class_maps.append(R_vals * self.gmm_proba[:,l].reshape(self.R_Nx, self.R_Ny))
        self.class_ims = class_maps
        return

    def spatial_separation(self, size, threshold = 0, method = 'yen', clean = True):
        """
        Identify spatially distinct regions from class images and separate based on a threshold and size.
        
        Args:
            size (int): number of pixels which is the minimum to keep a class - all spatially distinct regions with
                less than 'size' pixels will be removed
            threshold (float): intensity weight of a component to keep
            method (str): filter method, default 'yen'
            clean (bool): whether or not to 'clean' cluster sets based on overlap, i.e. remove clusters that do not have any unique components
        """
        labelled = []
        stacked = []
        cc_data = []
        for j in range(len(self.class_ims)):
            labelled_temp = []
            for l in range(len(self.class_ims[j])):
                image = np.where(self.class_ims[j][l] > threshold, 
                                self.class_ims[j][l], 0)
                if method == 'yen':
                    t=threshold_yen(image)
                elif method == 'otsu':
                    t = threshold_otsu(image)
                else:
                    print(method + ' method is not supported. Please use yen or otsu instead.')
                    break
                bw = closing(image > t, square(2))
                label_image = remove_small_objects(label(bw), size)
                unique_labels = np.unique(label_image)
                labelled_temp.extend([(np.where(
                    label_image == unique_labels[k],
                    image, 0)) for k in range(len(unique_labels))])

            if clean == False:
                labelled.append(labelled_temp)            
            elif clean == True:
                if len(labelled_temp) > 0:
                    stacked = np.dstack(labelled_temp)
                    data_hard = (stacked.max(axis=2,keepdims=1) == stacked) * stacked
                    data_list = [stacked[:,:,x] for x in range(stacked.shape[2])]
                    data_list_hard = [np.where(data_hard[:,:,n] > threshold, 1, 0) 
                                        for n in range(data_hard.shape[2])]
                    labelled.append([data_list[n] for n in range(len(data_list_hard)) 
                                    if (np.sum(data_list_hard[n]) > size)])
        if len(labelled_temp) > 0:
            self.spatially_separated_ims = labelled

        return
    
    def consensus(self, threshold = 0, location = 'spatially_separated_ims', method = 'mean', drop = 0, split = None):
        """
        Consensus Clustering takes the outcome of a prepared set of 2D images from each cluster and averages the outcomes.

        Args:
            threshold (float): Threshold weights, default 0
            location (str): Where to get the consensus from - after spatial separation = 'spatially_separated_ims'
            method (str): right now, mean is the only method in which to perform consensus clustering
            drop (int): number of clusters needed in each class to keep cluster set in the consensus. Default 0, meaning
            split (float): CURRENTLY NOT IMPLEMENTED - splitting threshold - if clusters in a consensus bin have less than the splitting threshold of overlap, create new bin
                no cluster sets will be dropped
        """
        class_dict = {}
        consensus_clusters = []
        if location == 'spatially_separated_ims':
            ncluster = [len(self.spatially_separated_ims[j]) 
                        for j in range(len(self.spatially_separated_ims))]
            max_cluster_ind = np.where(ncluster == np.max(ncluster))[0][0]

        for k in range(len(self.spatially_separated_ims[max_cluster_ind])):   
            class_dict['c'+str(k)] = [np.where(
                self.spatially_separated_ims[max_cluster_ind][k] > threshold, 
                self.spatially_separated_ims[max_cluster_ind][k], 0)
                                        ] 
        for j in range(len(self.spatially_separated_ims)):
            if j == max_cluster_ind:
                continue
            for m in range(len(self.spatially_separated_ims[j])):
                class_im = np.where(
                    self.spatially_separated_ims[j][m] > threshold, 
                    self.spatially_separated_ims[j][m], 0)
                best_sum = -np.inf
                for l in range(len(class_dict.keys())):
                    #if l >= len(self.spatially_separated_ims[j]):
                    #    break
                    current_sum = np.sum(np.where(
                        class_dict['c'+str(l)][0] > threshold, class_im, 0))
                    if current_sum >= best_sum:
                        best_sum = current_sum
                        cvalue = l
                if best_sum > 0:
                    class_dict['c' + str(cvalue)].append(class_im)
                else:
                    class_dict['c' + str(len(list(class_dict.keys())))] = [class_im]
                class_dict['c'+str(cvalue)].append(class_im)
            key_list = list(class_dict.keys())
            
        if method == 'mean':
            for n in range(len(key_list)):
                if drop > 0:
                    if len(class_dict[key_list[n]]) <= drop:
                        continue
                consensus_clusters.append(np.mean(np.dstack(
                    class_dict[key_list[n]]), axis = 2))
        self.consensus_dict = class_dict
        self.consensus_clusters = consensus_clusters
        return


@ignore_warnings(category=ConvergenceWarning)
def _nmf_single(
        x, 
        max_components,
        merge_thresh, 
        num_models, 
        max_iterations, 
        random_state=None, 
        save_all_models = True
    ):
    """
    Performs NMF on single feature matrix, which is an nd.array
    
    Args:
        x (np.ndarray):
        max_components (int): number of initial components to start the first NMF iteration
        merge_thresh (float): correlation threshold to merge features
        num_models (int): Number of independent models to run (number of learners that will be combined in consensus)
        iterations (int): number of iterations. Default 1, which runs traditional NMF
        random_state (int): random state
        save_all_models (bool): Whether or not to return all of the models - default is to save
            all outputs for consensus clustering
    """
    #Prepare error, random seed
    err = np.inf    
    if random_state == None:
        rng = np.random.RandomState(seed = 42)
    else:
        seed = random_state
    if save_all_models == True:
        W = []
        
    #Big loop through all models
    for i in range(num_models):
        if random_state == None:
            seed = rng.randint(5000)
        n_comps = max_components
        recon_error, counter = 0, 0
        Hs, Ws = [], []
        
        #Inner loop for iterative NMF
        for z in range(max_iterations):
            nmf = NMF(n_components = n_comps, random_state = seed)

            if counter == 0:
                nmf_temp = nmf.fit_transform(x)
            else:
                with np.errstate(invalid='raise',divide='raise'):
                    try:
                        nmf_temp_2 = nmf.fit_transform(nmf_temp)
                    except FloatingPointError:
                        print('Warning encountered in NMF: Returning last result')
                        break
            Ws.append(nmf_temp)
            Hs.append(np.transpose(nmf.components_))
            recon_error += nmf.reconstruction_err_
            counter += 1
            if counter > 1:
                with np.errstate(invalid='raise',divide='raise'):
                    try:
                        tril = np.tril(np.corrcoef(nmf_temp_2, rowvar = False), k = -1)
                        nmf_temp = nmf_temp_2
                    except FloatingPointError:
                        print('Warning encountered in correlation: Returning last result. Try larger merge_thresh.')
                        break
            else:
                tril = np.tril(np.corrcoef(nmf_temp, rowvar = False), k = -1)
                
            #Merge correlated features
            if np.nanmax(tril) >= merge_thresh:
                inds = np.argwhere(tril >= merge_thresh)
                for n in range(inds.shape[0]):
                    nmf_temp[:, inds[n,0]] += nmf_temp[:,inds[n,1]]
                ys_sorted = np.sort(np.unique(inds[n,1]))[::-1]
                for n in range(ys_sorted.shape[0]):
                    nmf_temp = np.delete(nmf_temp, ys_sorted[n], axis=1)
            else:
                break
            n_comps = nmf_temp.shape[1] - 1
            if n_comps <= 2:
                break
       
        if save_all_models == True:
            W.append(nmf_temp)

        elif (recon_error / counter) < err:
            err = (recon_error / counter)
            W = nmf_temp
    return W

@ignore_warnings(category=ConvergenceWarning)
def _gmm_single(x, cv, components, num_models, random_state=None, return_all=True):
    """
    Runs GMM several times and saves value with best BIC score
    
    Args:
        keys (list of str): List of strings associated with features
        cv (list of str): covariance
        components (list of ints)
        num_models (int)
        
        Returns:
        best_gmm
        best_gmm_labels
        best_gmm_proba
    """
    if return_all == True:
        gmm_list = []
        gmm_labels = []
        gmm_proba = []
    lowest_bic = np.infty
    bic = []
    bic_temp = 0
    if random_state == None:
        rng = np.random.RandomState(seed = 42)
    else:
        seed = random_state
    for n in range(num_models):
        if random_state == None:
            seed = rng.randint(5000)
        for j in range(len(components)):
            for cv_type in cv:
                gmm = GaussianMixture(n_components=components[j],
                                      covariance_type=cv_type, random_state = seed)
                labels = gmm.fit_predict(x)
                bic_temp = gmm.bic(x)    
        if return_all == True:
            gmm_list.append(gmm)
            gmm_labels.append(labels)
            gmm_proba.append(gmm.predict_proba(x))
        elif bic_temp < lowest_bic:
            lowest_bic = bic_temp
            best_gmm = gmm
            best_gmm_labels = labels
            best_gmm_proba = gmm.predict_proba(x)
    if return_all == True:
        return gmm_list, gmm_labels, gmm_proba
    else:
        return best_gmm, best_gmm_labels, best_gmm_proba
