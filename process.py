################################################################################
# CONTINUOUS GEOMETRY-AWARE DL-ROM
# Definition of classes pertaining to data handling and preprocessing
#
# Authors:  Simone Brivio, Stefania Fresca, Andrea Manzoni
# Date:     July 2023
################################################################################

import numpy as np
from utils import zeropad, TestCase


seed = 1111
np.random.seed(seed)


class DataProcessor:
    """ Loads and processes data.
    """

    def __init__(self, test_case : TestCase):
        """

        Args:
            test_case (testCase): the test case related to the data we want to
                                  load and process.
        """
        self.test_case = test_case
        self.config = self.test_case.config
        self.normalizer = None



    def loaddata(self):
        """ Load the data for the chosen test case.
        """

        filenames = self.test_case.filenames
        self.raw_data = dict()

        # Load npy files
        for key in filenames.keys():
             self.raw_data[key] = np.load(filenames[key], allow_pickle=True)
             if filenames[key].endswith('.npz'):
                 self.raw_data[key] = self.raw_data[key]['arr_0']

        # Get info about dofs
        dofs = [
            np.prod(self.raw_data['S'][i].shape) 
            for i in range(len(self.raw_data['S']))
        ]
        self.dofs_info = dict()
        self.dofs_info['max'] = np.max(dofs)
        self.dofs_info['min'] = np.min(dofs)
        self.N_data = self.raw_data['S'].shape[0]

        # Pad and reshape
        self.raw_data['S'], self.raw_data['mask'] = zeropad(self.raw_data['S'])
        self.raw_data['X'], _ = zeropad(self.raw_data['X'])
        for key in ('S', 'X'):
            if len(self.raw_data[key].shape) == 2:
                self.raw_data[key] = self.raw_data[key][:,:,None]



    def preprocess(self):
        """ Pre-process the loaded data
        """

        # Extract info from raw data and configuration file
        N_train = self.config.get('N_train')
        N_valid = self.config.get('N_valid')
        N_test = self.config.get('N_test')
        alpha_train = self.config.get('alpha_train')
        alpha_valid = self.config.get('alpha_train')
        alpha_test = self.config.get('alpha_test')
        N_t = self.config.get('N_t')
        if N_t is None:
            N_t = 1
        if (N_train is None) and (N_test is None):
            if (alpha_train is None) and (alpha_test is None):
                alpha_train = 0.8
                alpha_test = 0.1
            N_train = int(alpha_train * (self.N_data / N_t)) * N_t
            N_test = int(alpha_test * (self.N_data / N_t)) * N_t
        if N_valid is None and  alpha_valid is None:
            N_valid = self.N_data - N_train - N_test
        if N_valid is None:
            N_valid = int(alpha_valid * (self.N_data / N_t)) * N_t

        # Check coherency
        assert N_train + N_valid + N_test <= self.N_data

        # Initialize datasets
        self.train_data = dict()
        self.val_data = dict()
        self.test_data = dict()

        # Splitting raw data
        for key in self.raw_data.keys():
            if key is not None:
                self.train_data[key] = self.raw_data[key][:N_train]
                self.val_data[key] = self.raw_data[key][N_train:N_train+N_valid]
                self.test_data[key] = self.raw_data[key][-N_test:]
        if self.config.get('normalize') == True:
            self.normalizer = Normalizer(self.train_data)

        # Delete unnecessary raw data
        del self.raw_data
        
    



class Normalizer:
    """Performs the forward and the backward normalizer pass 
    """

    def __init__(self, train_data):
        """
        
        Args:
            train_data: the training data to compute min-max scaling constants.     
        """
        self.min = dict()
        self.max = dict()
        if train_data.get('mu') is not None:
            self.min['mu'] = np.min(train_data['mu'], axis = 0).astype('float32')
            self.max['mu'] = np.max(train_data['mu'], axis = 0).astype('float32')
        self.min['g'] = np.min(train_data['g'], axis = 0).astype('float32')
        self.max['g'] = np.max(train_data['g'], axis = 0).astype('float32')
        self.min['X'] = np.min(train_data['X'], axis = (0,1)).astype('float32')
        self.max['X'] = np.max(train_data['X'], axis = (0,1)).astype('float32')
        self.min['S'] = np.min(train_data['S'], axis = (0,1)).astype('float32')
        self.max['S'] = np.max(train_data['S'], axis = (0,1)).astype('float32')
    


    def forward_item(self, item, item_min, item_max):
        """ Forward pass per item.
        
        Args:
            item: the current item.
            item_min: the min scaling constant.
            item_max: the max scaling constant.
        
        Return:
            the scaled item
        """
        return (item - item_min) / (item_max - item_min)



    def backward_item(self, item, item_min, item_max):
        """ Backward pass per item.
        
        Args:
            item: the current item.
            item_min: the min scaling constant.
            item_max: the max scaling constant.
        
        Return:
            the unscaled item.
        """
        return item_min + item * (item_max - item_min)



    def forward(self, data):
        """ Forward pass.

        Args:
            data: input data.
        
        Return:
            scaled data.
        """
        for key in data.keys():
            data[key] = self.forward_item(
                item = data[key], 
                item_min = self.min[key].flatten(), 
                item_max = self.max[key].flatten()
            )
        return data
    


    def backward(self, output):
        """ Backward pass.

        Args:
            data: input data.
        
        Return:
            unscaled data.
        """
        output = self.backward_item(output, 
                                    self.min['S'].flatten(), 
                                    self.max['S'].flatten())
        return output
