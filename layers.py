################################################################################
# CONTINUOUS GEOMETRY-AWARE DL-ROM
# Implementation of neural network layers and layer utilities
#
# Authors:  Simone Brivio, Stefania Fresca, Andrea Manzoni
# Date:     July 2023
################################################################################

import numpy as np
from collections.abc import Callable

# Import GPU-based libraries
import ctypes
_libcudart = ctypes.CDLL('libcudart.so')
# Set device limit on the current device
# cudaLimitMaxL2FetchGranularity = 0x05
pValue = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
_libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
_libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
assert pValue.contents.value == 128
import os
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from nns import *
from utils import pod_basis



class CGAProjLayer(tf.keras.Model):
    """ Performs the projection at a continuous level thanks to the global
        basis function given as argument.
    """


    def __init__(
        self, 
        global_basis_layer : tf.keras.Model, 
        name : str = 'cont_proj'
    ):
        """

        Args:
            global_basis_layer (tf.keras.Model): used to compute the CGA basis
                                                 functions.
            name (str): the layer name (defaults to 'cont_proj').
        """
        super(CGAProjLayer, self).__init__(name = name)
        self.global_basis_layer = global_basis_layer
        self.flatten = tf.keras.layers.Flatten()


    def call(self, x, u_hf, g, training = False):
        """ Performs the projection.

        Args:
            x: input locations.
            u_hf: input values.
            g: geometrical parameters.
            training: True during training (defaults to False).
        """
        basis = self.global_basis_layer(x, g)
        coeffs = self.flatten(tf.einsum('bijc,bic->bjc', basis, u_hf))
        return coeffs
 




class CGALiftLayer(tf.keras.Model):
    """ Performs the lifting at a continuous level thanks to the global
        basis function given as argument
    """


    def __init__(
        self, 
        global_basis_layer : tf.keras.Model,
        name = 'cont_lift'
    ):
        """

        Args:
            global_basis_layer (tf.keras.Model): used to compute the CGA basis
                                                 functions.
            name (str): the layer name (defaults to 'cont_lift').
        """
        super(CGALiftLayer, self).__init__(name = name)
        self.global_basis_layer = global_basis_layer


    def call(self, x, u_encoded, g, training = False):
        """ Performs the lifting.

        Args:
            x: output locations.
            u_encoded: computed coefficients.
            g: geometrical parameters.
            training: True during training (defaults to False).
        """
        basis =  self.global_basis_layer(x, g)
        lift_pred = tf.einsum('bijc,bjc->bic', basis, u_encoded)
        return lift_pred
  




class PODProjLayer(tf.keras.Model):
    """ Performs the projection at discrete level thanks to the precomputed POD 
        matrix
    """


    def __init__(
        self, 
        train_data, 
        N : int, 
        name : str = 'discr_proj'
    ):
        """

        Args:
            train_data: the labeled training data.
            N: the reduced dimension.
            name (str): the layer name (defaults to 'discr_lift').
        """
        super(PODProjLayer, self).__init__(name = name)
        n_channels = train_data.shape[2]
        self.flatten = tf.keras.layers.Flatten()
        basis = [
            pod_basis(train_data[:,:,c_idx].T, N) for c_idx in range(n_channels)
        ]
        self.basis = np.array(basis).transpose((1,2,0))


    def call(self, u_hf, training = False):
        """ Performs the projection.

        Args:
            u_hf: input values.
            training: True during training (defaults to False).
        """
        coeffs = tf.einsum('ijc,bic->bjc', self.basis, u_hf)
        return self.flatten(coeffs)





class PODLiftLayer(tf.keras.Model):
    """ Performs the lifting at discrete level thanks to the precomputed POD 
        matrix
    """


    def __init__(self, basis, name = 'discr_lift'):
        """

        Args:
            basis: the POD basis
            name (str): the layer name (defaults to 'discr_lift').
        """
        super(PODLiftLayer, self).__init__(name = name)
        self.basis = basis


    def call(self, u_encoded, training = False):
        """ Performs the projection.

        Args:
            u_encoded: computed coefficients.
            training: True during training (defaults to False).
        """
        lift_pred = tf.einsum('ijc,bjc->bic', self.basis, u_encoded)
        return lift_pred





class CGAbasis(tf.keras.Model):
    """ An instance of geometry-dependent global basis functions enhanced 
        by a predefined feature engineering
    """


    def __init__(
        self,
        ffnn : tf.keras.Model, 
        c : int, 
        feature_engineering : Callable = None
    ):
        """

        Args:
            ffnn (tf.keras.Model): the (feed-forward) neural network.
            c (int): number of channels.
            feature_engineering: function that performs feature engineering.
        """
        super(CGAbasis, self).__init__()
        self.ffnn = ffnn
        self.feature_engineering = feature_engineering
        self.c = c


    def call(self, x, g, training = False):
        """ Performs the forward pass.

        Args:
            x: output locations.
            g: geometrical parameters.
            training: True during training (defaults to False).
        """
        if self.feature_engineering is not None:
            x = self.feature_engineering(x)
        x = tf.concat(
            (x, tf.repeat(g[:,None,:], axis = 1, repeats = tf.shape(x)[1])), 
            axis = 2
        )
        x = self.ffnn(x)
        return x
    

