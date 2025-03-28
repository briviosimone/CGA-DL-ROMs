################################################################################
# CONTINUOUS GEOMETRY-AWARE DL-ROM
# Implementation of some utilities
#
# Authors:  Simone Brivio, Stefania Fresca, Andrea Manzoni
# Date:     July 2023
################################################################################

import sys
import abc
import copy
import numpy as np
import matplotlib.pyplot as plt
import pickle


seed = 1111
np.random.seed(seed)


class TestCase(abc.ABC): 
    """ Test case base class.
    """

    def __init__(
        self, 
        config : dict,
        save_folder,
        filenames
    ):
        """ 
        
        Args:
            config: the test case config dictionary.
            save_folder: the save folder.
            filenames: the data filenames.
        """

        self.config = config
        self.save_folder = save_folder
        self.filenames = filenames
    
    
    @abc.abstractmethod
    def postprocess(self):
        pass
    


def get_class_name(self):
    """ Utility for getting class name. 
    """
    return self.__class__.__name__



def get_filename(model, save_folder, file_id, file_extension):
    """ Utility for getting post-processing filename with a convention. 
    """
    class_name = get_class_name(model).lower()
    name = save_folder
    if model.model_name != '':
        name = name + model.model_name + "_" 
    name = name + class_name + file_id + file_extension
    return name



def savefig_predictions(model, save_folder : str, idx : str = ''):
    """ Utility for saving prediction figures.

    Args:
        model: the NN model.
        save_folder (str): the save folder.
        idx (str): the sample number (defaults to '')
    """
    filename = get_filename(
        model = model, 
        save_folder = save_folder,
        file_id = '_sample' + str(idx),
        file_extension = '.jpg'
    )
    plt.savefig(filename)



def savemp4_predictions(model, anim, fps, save_folder : str, idx = ''):
    """ Utility for saving prediction animations.
    
    Args:
        model: the NN model.
        anim: matplotlib animation object.
        fps: frame per seconds.
        save_folder (str): the save folder.
        idx (str): the sample number (defaults to '')
    """
    filename = get_filename(
        model = model, 
        save_folder = save_folder,
        file_id = '_sample' + str(idx),
        file_extension = '.gif'
    )
    anim.save(
        filename = filename, 
        writer='pillow', 
        fps = fps, 
        bitrate = 30
    )



def zeropad(input : np.array):
    """ Zero-padding for numpy arrays.

    Args:
        input (np.array): input numpy array.

    Return:
        Padded input, mask.
    """
    max_len = np.max([item.shape[0] for item in input])
    if len(input[0].shape) < 2:
        output = np.zeros((input.shape[0], max_len))
    else:
        output = np.zeros((input.shape[0], max_len, input[0].shape[1]))
    mask = np.zeros((input.shape[0], max_len), dtype = bool)
    for i in range(input.shape[0]):
        curr_len = input[i].shape[0]
        output[i,:curr_len] = input[i]
        mask[i,:curr_len] = True
    return output, mask   



def plot_history(model, history, save_folder):
    """ Utility to plot the training history.

    Args:
        model: the NN model.
        history: the training history.
        save_folder (str): the save folder.
    """
    filename = get_filename(
        model = model, 
        save_folder = save_folder,
        file_id = '_history',
        file_extension = '.jpg'
    )
    plt.figure()
    plt.semilogy(history.history['loss'], label = 'train')
    plt.semilogy(history.history['val_loss'], label = 'valid')
    plt.legend()
    plt.savefig(filename)



def save_history(model, history, save_folder):
    """ Utility to save the training history onto a PICKLE file.

    Args:
        model: the NN model.
        history: the training history.
        save_folder (str): the save folder.
    """
    filename = get_filename(
        model = model, 
        save_folder = save_folder,
        file_id = '_history',
        file_extension = '.pickle'
    )
    with open(filename, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)



def relative_error_componentwise(y_true, y_pred, L2 = False):
    """ Computes the relative error for a single component.

    Args:
        y_true: targets.
        y_pred: predictions.
        L2 (bool): if True, we use the L^2(P) norm (defaults to False).

    Return:
        The computed metric.
    """

    if not L2: 
        num = np.linalg.norm(y_pred - y_true, axis = 1)
        den = np.linalg.norm(y_true, axis = 1)
        test_err = np.mean(num / den)
    else:
        num = np.sqrt(np.mean(np.sum((y_pred - y_true)**2, axis = 1)))
        den = np.sqrt(np.mean(np.sum(y_true**2, axis = 1)))
        test_err = num / den
    return test_err



def relative_error(y_true, y_pred, L2 = False):
    """ Computes the component-averaged relative error.

    Args:
        y_true: targets.
        y_pred: predictions.
        L2 (bool): if True, we use the L^2(P) norm (defaults to False).

    Return:
        The computed metric.
    """
    err = list()
    for c_idx in range(y_true.shape[2]):
        curr_err = relative_error_componentwise(
            y_true[:,:,c_idx],
            y_pred[:,:,c_idx],
            L2
        )
        err.append(curr_err)
    err = np.mean(np.array(err))
    return err

    

def pod_basis(mat, N : int):
    """ Utility to compute the (truncated) POD basis with SVD.

    Args:
        mat: the snapshot matrix.
        N (int): the POD dimension.
    
    Return:
        The POD basis.
    """
    U, _, _ = np.linalg.svd(
        mat.astype('float32'), 
        full_matrices = False, 
        compute_uv = True
    )
    return U[:,:N]



def pod_error_componentwise(mat, basis):
    """ Utility to compute the POD error for a single component.

    Args:
        mat: the test snapshot matrix.
        basis: the POD basis.
    
    Return:
        the POD error.
    """
    projected_mat = basis @ basis.T @ mat
    num = np.sqrt(np.mean(np.sum((mat.T - projected_mat.T)**2, axis = 1)))
    den = np.sqrt(np.mean(np.sum((mat.T)**2, axis = 1)))
    pod_err = num / den
    return pod_err



def pod_error(mat_train, mat_test, N):
    """ Utility to compute the component-averaged POD error.

    Args:
        mat_train: the train snapshot matrix.
        mat_test: the test snapshot matrix.
        N (int): the POD dimension.
    
    Return:
        The component-averaged POD error.
    """
    pod_err = list()
    c = mat_train.shape[2]
    for c_idx in range(c):
        basis = pod_basis(mat_train[:,:,c_idx].T, N)
        pod_err_componentwise = pod_error_componentwise(
            mat_test[:,:,c_idx].T, basis
        )
        pod_err.append(pod_err_componentwise)
    pod_err = np.mean(np.array(pod_err))
    return pod_err
    
    

class DofsSubsampler:
    """ Subsamples DOFS with a random criterion. 
    """

    def random_subsample_filter(self, ratio : float):
        """ Creates subsample index filter.

        Args:
            ratio (float): the subsample ratio.
        """
        n_sub = int(ratio * self.n_samples)
        return self.subsample_filter[:n_sub]



    def apply(self, data, ratio : float):
        """ Applies subsampling to given data.

        Args:
            data: the given data.
            ratio (float): the subsample ratio.
        
        """

        # Random sample indices and create subsample filter
        if not hasattr(self, 'subsample_filter'):
            self.n_samples = data['X'].shape[1]
            self.subsample_filter = np.arange(self.n_samples)
            np.random.shuffle(self.subsample_filter)
        curr_subsample_filter = self.random_subsample_filter(ratio)

        # Apply created filter
        mod_data = copy.copy(data)
        for key in ('S', 'X', 'mask'):
            mod_data[key] = mod_data[key][:,curr_subsample_filter]

        return mod_data
