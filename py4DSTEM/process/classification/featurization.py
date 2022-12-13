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
    array of input features for each pattern. Features are stored under Featurization. Features and can be
    used for a variety of unsupervised classification tasks.
    
    Initialization methods:
        __init__:
            Creates instance of featurization.
    
    Feature Dictionary Modification Methods
        add_feature:
            Adds features to the features array
        remove_feature:
            Removes features to the features array
    
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
            Performs either traditional or iterative Nonnegative Matrix Factorization (NMF) to refine features.

        GMM:
            Gaussian mixture model to predict class labels. Fits a gaussian based on covariance of features.
    
    Class Examination Methods
        get_class_DPs:
            Gets weighted class diffraction patterns (DPs) for an NMF or GMM operation
        get_class_ims:
            Gets weighted class images (ims) for an NMF or GMM operation
    """
    
    def __init__(self, features, R_Nx, R_Ny, name):
        """
        Initializes classification instance.
        
        This method:
        1. Generates key:value pair to access input features
        2. Initializes the empty dictionaries for feature modification and classification
        
        Args:
            features (list):    A list of ndarrays which will each be associated with value stored at the key in the same index within the list
            R_Nx (int):         The real space x dimension of the dataset
            R_Ny (int):         The real space y dimension of the dataset
            name (str):         The name of the featurization object
        """
        self.R_Nx = R_Nx
        self.R_Ny = R_Ny
        self.name = name

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
            if all(isinstance(f, np.ndarray) for f in features):
                for i in range(len(features)):
                    if features[i].shape == 3:
                        features[i] = features[i].reshape(R_Nx*R_Ny, features.shape[-1])
                    if len(features[i].shape) != 2:
                        raise ValueError(
                            'feature array(s) in list must be of dimensions (R_Nx*R_Ny, num_features) or (R_Nx, R_Ny, num_features)'
                            )  
                self.features = np.concatenate(features, axis=1)
            elif all(isinstance(f, Featurization) for f in features):
                raise TypeError('List of Featurization instances must be initialized using the concatenate_features method.')
            else:
                raise TypeError('Entries in list must be np.ndarrays for initialization of the Featurization instance.') 
        else:
            raise TypeError('Features must be either a single np.ndarray of shape 2 or 3 or a list of np.ndarrays or featurization instances.')
        return

    def concatenate_features(features, name):
        """
        Concatenates featurization instances (features) and outputs a new Featurization instance
        containing the concatenated features from each featurization instance. R_Nx, R_Ny will be
        inherited from the featurization instances and must be consistent across objects.
        
        Args:
            features (list):    A list of keys to be concatenated into one array
            name (str):         The name of the featurization instance
        """
        R_Nxs = [features[i].R_Nx for i in range(len(features))]
        R_Nys = [features[i].R_Ny for i in range(len(features))]
        if len(np.unique(R_Nxs)) != 1 or len(np.unique(R_Nys)) != 1:
            raise ValueError('Can only concatenate Featurization instances with same R_Nx and R_Ny')
        new_instance = Featurization(
                np.concatenate([features[i].features for i in range(len(features))], axis = 1),
                R_Nx = R_Nxs[0],
                R_Ny = R_Nys[0],
                name = name
            )
        return new_instance
    
    def add_features(self, feature):
        """
        Add a feature to the end of the features array
        
        Args:
            key (int, float, str):  A key in which a feature can be accessed from
            feature (ndarray):      The feature associated with the key
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
    
    def mean_feature(self, index):
        """
        Takes columnwise mean and replaces features in 'index'.
        
        Args:
            index (list of int): Indices of features to take the mean of. New feature array is placed in self.features.
        """
        mean_features = np.mean(self.features[:,index], axis = 1)
        mean_features = mean_features.reshape(mean_features.shape[0], 1)
        cleaned_features = np.delete(self.features, index, axis=1)
        self.features = np.concatenate([cleaned_features, mean_features], axis=1)
        return

    def median_feature(self, index):
        """
        Takes columnwise median and replaces features in 'index'. New feature array is placed in self.features.
        
        Args:
            index (list of int): Indices of features to take the median of.
        """
        median_features = np.median(self.features[:,index], axis = 1)
        median_features = median_features.reshape(median_features.shape[0], 1)
        cleaned_features = np.delete(self.features, index, axis=1)
        self.features = np.concatenate([cleaned_features, median_features], axis=1)
        return
    
    def max_feature(self, index):
        """
        Takes columnwise max and replaces features in 'index'. New feature array is placed in self.features.
        
        Args:
            index (list of int): Indices of features to take the max of.
        """
        max_features = np.max(self.features[:,index], axis = 1)
        max_features = max_features.reshape(max_features.shape[0], 1)
        cleaned_features = np.delete(self.features, index, axis=1)
        self.features = np.concatenate([cleaned_features, max_features], axis=1)
        return

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
    
    def PCA(self, components, return_results = False):
        """
        Performs PCA on features
        
        Args:
            components (list): A list of ints for each key. This will be the output number of features
        """
        pca = PCA(n_components = components)
        self.pca = pca.fit_transform(self.features)
        if return_results == True:
            return self.pca
        return

    def ICA(self, components, return_results = True):
        """
        Performs ICA on features
        
        Args:
            components (list): A list of ints for each key. This will be the output number of features
        """
        ica = FastICA(n_components = components)
        self.ica = ica.fit_transform(self.features)
        if return_results == True:
            return self.ica
        return
    
    def NMF(
        self,
        max_components, 
        num_models, 
        merge_thresh = 1,
        max_iterations = 1, 
        random_seed = None, 
        save_all_models = True,
        return_results = False
    ):
        """
        Performs either traditional Nonnegative Matrix Factoriation (NMF) or iteratively on input features.
        For Traditional NMF:
            set either merge_threshold = 1, max_iterations = 1, or both. Default is to set 
        
        Args:
            max_components (int):   Number of initial components to start the first NMF iteration
            merge_thresh (float):   Correlation threshold to merge features
            num_models (int):       Number of independent models to run (number of learners that will be combined in consensus).
            max_iterations (int):   Number of iterations. Default 1, which runs traditional NMF
            random_seed (int):      Random seed.
            save_all_models (bool): Whether or not to return all of the models - default is to return all outputs for consensus clustering.
                                        if False, will only return the model with the lowest NMF reconstruction error.
            return_results (bool):  Whether or not to return the final class weights
        
        Details:
            This method may require trial and error for proper selection of parameters. To perform traditional NMF, the 
            defaults should be used:
                merge_thresh = 1
                max_iterations = 1
            Note that the max_components in this case will be equivalent to the number of classes the NMF model identifies.
            
            Iterative NMF calculates the correlation between all of the output columns from an NMF iteration, merges the
            features correlated above the merge_thresh, and performs NMF until either max_iterations is reached or until
            no more columns are correlated above merge_thresh.
        """
        self.W = _nmf_single(
            self.features,
            max_components=max_components,
            merge_thresh = merge_thresh,
            num_models = num_models,
            max_iterations = max_iterations, 
            random_seed = random_seed,
            save_all_models = save_all_models
        )
        if return_results == True:
            return self.W
        return
        
    def GMM(self, cv, components, num_models, random_seed = None, return_results = False):
        """
        Performs gaussian mixture model on input features
        
        Args:
            cv (str):           Covariance type - must be 'spherical', 'tied', 'diag', or 'full'
            components (int):   Number of components
            num_models (int):   Number of models to run
            random_seed (int): Random seed
        """
        self.gmm, self.gmm_labels, self.gmm_proba = _gmm_single(
            self.features,
            cv=cv,
            components = components,
            num_models = num_models, 
            random_seed = random_seed
        )
        if return_results == True:
            return self.gmm
        return

    def get_class_DPs(self, datacube, classification_method, thresh):
        """
        Returns weighted class patterns based on classification instance
        datacube must be vectorized in real space (shape = (R_Nx * R_Ny, 1, Q_Nx, Q_Ny)
        
        Args:
            classification_method (str):    Either 'nmf' or 'gmm' - finds location of clusters
            datacube (py4DSTEM datacube):   Vectorized in real space, with shape (R_Nx * R_Ny, Q_Nx, Q_Ny)
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
            if type(self.gmm_labels) == list:
                for l in range(len(self.gmm_labels)):
                    small_class_maps = []
                    for k in range(np.max(self.gmm_labels[l])):
                        R_vals = np.where(self.gmm_labels[l].reshape(self.R_Nx, self.R_Ny) == k, 1, 0)
                        small_class_maps.append(R_vals * self.gmm_proba[l][:,k].reshape(self.R_Nx, self.R_Ny))
                    class_maps.append(small_class_maps)
            else:
                for l in range((np.max(self.gmm_labels))):
                    R_vals = np.where(self.gmm_labels[l].reshape(self.R_Nx,self.R_Ny) == l, 1, 0)
                    class_maps.append(R_vals * self.gmm_proba[:,l].reshape(self.R_Nx, self.R_Ny))
        self.class_ims = class_maps
        return

    def spatial_separation(
            self,
            size,
            threshold = 0,
            method = None,
            clean = True
        ):
        """
        Identify spatially distinct regions from class images and separate based on a threshold and size.
        
        Args:
            size (int):         Number of pixels which is the minimum to keep a class - all spatially distinct regions with
                                    less than 'size' pixels will be removed
            threshold (float):  Intensity weight of a component to keep
            method (str):       (Optional) Filter method, default None. Accepts options 'yen' and 'otsu'.
            clean (bool):       Whether or not to 'clean' cluster sets based on overlap, i.e. remove clusters that do not have 
                                    any unique components
        """
        #Prepare for separation
        labelled = []
        stacked = []
        
        #Loop through all models
        for j in range(len(self.class_ims)):
            separated_temp = []
            
            #Loop through class images in each model to filtered and separate class images
            for l in range(len(self.class_ims[j])):
                image = np.where(self.class_ims[j][l] > threshold, 
                                self.class_ims[j][l], 0)
                if method == 'yen':
                    t = threshold_yen(image)
                    bw = closing(image > t, square(2))
                    labelled_image = label(bw)
                    if np.sum(labelled_image) > size:
                        large_labelled_image = remove_small_objects(labelled_image, size)
                    else:
                        large_labelled_image = labelled_image
                elif method == 'otsu':
                    t = threshold_otsu(image)
                    bw = closing(image > t, square(2))
                    labelled_image = label(bw)
                    if np.sum(labelled_image) > size:
                        large_labelled_image = remove_small_objects(labelled_image, size)
                    else:
                        large_labelled_image = labelled_image
                elif method == None:
                    labelled_image = label(image)
                    if np.sum(labelled_image) > size:
                        large_labelled_image = remove_small_objects(labelled_image, size)
                    else:
                        large_labelled_image = labelled_image
                    
                else:
                    print(method + ' method is not supported. Please use yen, otsu, or None instead.')
                    break
                unique_labels = np.unique(large_labelled_image)
                separated_temp.extend(
                    [(np.where(large_labelled_image == unique_labels[k+1],image, 0)) 
                    for k in range(len(unique_labels)-1)
                    ])
            
            if len(separated_temp) > 0:
                if clean == True:
                    data_ndarray = np.dstack(separated_temp)
                    data_hard = (data_ndarray.max(axis=2,keepdims=1) == data_ndarray) * data_ndarray
                    data_list = [data_ndarray[:,:,x] for x in range(data_ndarray.shape[2])]
                    data_list_hard = [np.where(data_hard[:,:,n] > threshold, 1, 0) 
                                        for n in range(data_hard.shape[2])]
                    labelled.append([data_list[n] for n in range(len(data_list_hard)) 
                                    if (np.sum(data_list_hard[n]) > size)])            
                else:
                    labelled.append(separated_temp)
            else:
                continue
                 
        if len(labelled) > 0:
            self.spatially_separated_ims = labelled
        else:
            print('No distinct regions found in any models. Try modifying threshold, size, or method.')

        return
    
    def consensus(
            self, 
            threshold = 0, 
            location = 'spatially_separated_ims', 
            split = 0,
            method = 'mean', 
            drop_bins= 0, 
            
        ):
        """
        Consensus Clustering takes the outcome of a prepared set of 2D images from each cluster and averages the outcomes.

        Args:
            threshold (float):      Threshold weights, default 0
            location (str):         Where to get the consensus from - after spatial separation = 'spatially_separated_ims'
            split_value (float):    Threshold in which to separate classes during label correspondence (Default 0). This should be 
                                        proportional to the expected class weights- the sum of the weights in the current class image 
                                        that match nonzero values in each bin are calculated and then checked for splitting.
            method (str):           Method in which to combine the consensus clusters - either mean or median.
            drop_bins (int):        Number of clusters needed in each class to keep cluster set in the consensus. Default 0, meaning
        
        Details:
            This method involves 2 steps: Label correspondence and consensus clustering. 
            
            Label correspondence sorts the classes found by the independent models into bins based on class overlap in real space. 
            Arguments related to label correspondence are the threshold and split_value. The threshold is related
            to the weights of the independent classes. If the weight of the observation in the class is less than the threshold, it
            will be set to 0. The split_value indicates the extent of similarity the independent classes must have before intializing
            a new bin. The default is 0 - this means if the class of interest has 0 overlap with the identified bins, a new bin will
            be created. The value is based on the sum of the weights in the current class image that match the nonzero values in the
            current bins.
            
            Consensus clustering combines these sorted bin into 1 class based on the selected method (either 'mean' which takes 
            the average of the bin, or 'median' which takes the median of the bin). Bins with less than the drop_bins value will 
            not be included in the final results.  
        """
        # Set up for consensus clustering
        class_dict = {}
        consensus_clusters = []
        
        if location != 'spatially_separated_ims':
            raise ValueError('Consensus clustering only supported for location = spatially_separated_ims.')
        
        #Find model with largest number of clusters for label correspondence
        ncluster = [len(self.spatially_separated_ims[j]) 
                    for j in range(len(self.spatially_separated_ims))]
        max_cluster_ind = np.where(ncluster == np.max(ncluster))[0][0]

        # Label Correspondence
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
                    self.spatially_separated_ims[j][m], 0
                )
                best_sum = -np.inf
                for l in range(len(class_dict.keys())):
                    current_sum = np.sum(np.where(
                        class_dict['c'+str(l)][0] > threshold, class_im, 0)
                    )
                    if current_sum >= best_sum:
                        best_sum = current_sum
                        cvalue = l
                if best_sum > split:
                    class_dict['c' + str(cvalue)].append(class_im)
                else:
                    class_dict['c' + str(len(list(class_dict.keys())))] = [class_im]
            key_list = list(class_dict.keys())
         
        #Consensus clustering   
        if method == 'mean':
            for n in range(len(key_list)):
                if drop_bins > 0:
                    if len(class_dict[key_list[n]]) <= drop_bins:
                        continue
                consensus_clusters.append(np.mean(np.dstack(
                    class_dict[key_list[n]]), axis = 2))
        elif method == 'median':
            for n in range(len(key_list)):
                if drop_bins > 0:
                    if len(class_dict[key_list[n]]) <= drop_bins:
                        continue
                consensus_clusters.append(np.median(np.dstack(
                    class_dict[key_list[n]]), axis = 2))
        else:
            print('Only mean and median consensus methods currently supported.')    
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
        random_seed=None, 
        save_all_models = True
    ):
    """
    Performs NMF on single feature matrix, which is an nd.array
    
    Args:
        x (np.ndarray):         Feature array
        max_components (int):   Number of initial components to start the first NMF iteration
        merge_thresh (float):   Correlation threshold to merge features
        num_models (int):       Number of independent models to run (number of learners that will be combined in consensus)
        iterations (int):       Number of iterations. Default 1, which runs traditional NMF
        random_seed (int):      Random seed
        save_all_models (bool): Whether or not to return all of the models - default is to save
                                    all outputs for consensus clustering
    """
    #Prepare error, random seed
    err = np.inf    
    if random_seed == None:
        rng = np.random.RandomState(seed = 42)
    else:
        seed = random_seed
    if save_all_models == True:
        W = []
        
    #Big loop through all models
    for i in range(num_models):
        if random_seed == None:
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
            if counter >= max_iterations:
                break
            elif counter > 1:
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
def _gmm_single(
        x,
        cv,
        components,
        num_models,
        random_seed=None,
        return_all=True
    ):
    """
    Runs GMM several times and saves value with best BIC score
    
    Args:
        x (np.ndarray):             Data
        cv (list of str):           Covariance, must be 'spherical', 'tied', 'diag', or 'full'
        components (list of ints):  Number of output clusters
        num_models (int):           Number of models to run. Only one is returned
        random_seed (int):          Random seed
        return_all (bool):          Whether or not to return all models.
        
        Returns:
        gmm_list OR best_gmm:           List of class identity or classes for best model
        gmm_labels OR best_gmm_labels:  Label list for all models or labels for best model
        gmm_proba OR best_gmm_proba:    Probability list of class belonging or probability for best model
    """
    if return_all == True:
        gmm_list = []
        gmm_labels = []
        gmm_proba = []
    lowest_bic = np.infty
    bic_temp = 0
    if random_seed == None:
        rng = np.random.RandomState(seed = 42)
    else:
        seed = random_seed
    for n in range(num_models):
        if random_seed == None:
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
                    
                elif return_all == False:
                    if (bic_temp < lowest_bic):
                        lowest_bic = bic_temp
                        best_gmm = gmm
                        best_gmm_labels = labels
                        best_gmm_proba = gmm.predict_proba(x)
            
    if return_all == True:
        return gmm_list, gmm_labels, gmm_proba
    return best_gmm, best_gmm_labels, best_gmm_proba
