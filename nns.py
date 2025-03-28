################################################################################
# CONTINUOUS GEOMETRY-AWARE DL-ROM
# Implementation of basic sample neural network blocks
#
# Authors:  Simone Brivio, Stefania Fresca, Andrea Manzoni
# Date:     July 2023
################################################################################

import numpy as np

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


# SET RANDOM SEED
seed = 1111
np.random.seed(seed)
tf.random.set_seed(seed)



class OrthogonalModInit(tf.keras.initializers.Initializer):
    """ Implements a "regularized" orthogonal initializer.
    """

    def __init__(self, delta = 1e-2):
        """ 
        
        Args:
            delta: regularization parameter (defults to 1e-2).
        """
        self.orthogonal = tf.keras.initializers.Orthogonal()
        self.identity = tf.keras.initializers.Identity(gain = delta)


    def __call__(self, shape, dtype=None):
        """ 
        
        Args:
            shape: Input tensor shape.
            dtype: Input tensor type.
        """
        init = self.orthogonal(shape, dtype=dtype) + \
            self.identity(shape, dtype=dtype)
        return init





class XavierModInit(tf.keras.initializers.Initializer):
    """ Implements a "regularized" Xavier initializer.
    """

    def __init__(self, delta = 1e-3):
        """ 
        
        Args:
            delta: regularization parameter (defults to 1e-3).
        """
        self.xavier = tf.keras.initializers.GlorotUniform()
        self.identity = tf.keras.initializers.Identity(gain = delta)


    def __call__(self, shape, dtype=None):
        """ 
        
        Args:
            shape: Input tensor shape.
            dtype: Input tensor type.
        """
        init = self.xavier(shape, dtype=dtype) + \
            self.identity(shape, dtype=dtype)
        return init





class DenseNetwork(tf.keras.Model):
    """ Implements a dense block of constant width and fixed depth.
    """

    def __init__(
        self, 
        width : int, 
        depth : int, 
        output_dim : int,
        activation = 'gelu', 
        kernel_initializer = OrthogonalModInit,
        residual_connections : bool = False
    ):
        """ 
        
        Args:
            width (int): the NN width.
            depth (int): the NN depth.
            output_dim (int): the output dimension.
            activation: the nonlinearity (defaults to 'gelu').
            kernel_initializer: for the weights initialization (defaults to 
                                OrthogonalModInit).
            residual_connections (bool): If True, perform residual connections
                                         in the hidden layers.
        """

        super(DenseNetwork, self).__init__()

        # Extract quantities
        self.width = width
        self.depth = depth
        self.output_dim = output_dim

        # Defines the first (depth - 1) layers
        self.dense_layers = [
            tf.keras.layers.Dense(self.width,
                                  activation = activation,
                                  kernel_initializer = kernel_initializer)
            for i in range(depth-1)
        ]

        # Defines the last layer
        self.dense_layers.append(
            tf.keras.layers.Dense(output_dim,
                                  activation = 'linear',
                                  kernel_initializer = kernel_initializer)
        )

        # Defines residual connection mechanism
        if residual_connections:
            self.hidden_mechanism = lambda layer, x: x + layer(x)
        else:
            self.hidden_mechanism = lambda layer, x: layer(x)
    


    def call(self, l, training = False):
        """ Forward pass function.

        Args:
            l: NN input.
            training: True during training (defaults to False).

        Returns:
            The NN output.
        """

        l = self.dense_layers[0](l)
        for i in range(1,self.depth - 1):
            l = self.hidden_mechanism(self.dense_layers[i], l)
        if self.depth > 1:
            l = self.dense_layers[self.depth-1](l)
        return l
    




class ConvBlock(tf.keras.Model):
    """ Implements a block of 4 convolutions.
    """

    def __init__(
        self, 
        input_image_size : int, 
        output_dim : int, 
        activation = 'gelu', 
        kernel_initializer = tf.keras.initializers.Orthogonal
    ):
        """
        
        Args:
            input_image_size (int): the input resolution.
            output_dim (int): the output dimension.
            activation: the nonlinearity (defaults to 'gelu').
            kernel_initializer: for the weights initialization (defaults to 
                                tf.keras.initializers.Orthogonal).
        """

        super(ConvBlock, self).__init__()

        # Dense layer at input and reshape to get 2D
        self.dense_in = tf.keras.layers.Dense(
            input_image_size**2, activation = activation, 
            kernel_initializer = kernel_initializer, name = 'dense_in')
        self.reshape = tf.keras.layers.Reshape(
            (input_image_size, input_image_size, 1), name = 'reshape1')
        
        # Hidden convolutional layers
        self.conv1 = tf.keras.layers.Conv2D(
            filters = 8, kernel_size = [5, 5], strides = (1, 1), 
            padding = 'same', activation = activation, 
            kernel_initializer = kernel_initializer, name = 'conv1')
        self.conv2 = tf.keras.layers.Conv2D(
            filters = 16, kernel_size = [5, 5], strides = (2, 2), 
            padding = 'same', activation = activation, 
            kernel_initializer = kernel_initializer, name = 'conv2')
        self.conv3 = tf.keras.layers.Conv2D(
            filters = 32, kernel_size = [5, 5], strides = (2, 2), 
            padding = 'same', activation = activation, 
            kernel_initializer = kernel_initializer, name = 'conv3')
        self.conv4 = tf.keras.layers.Conv2D(
            filters = 64, kernel_size = [5, 5], strides = (2, 2), 
            padding = 'same', activation = activation, 
            kernel_initializer = kernel_initializer, name = 'conv4')
        
        # Flatten to get 1D and dense layer at output
        self.flatten = tf.keras.layers.Flatten()
        self.dense_out = tf.keras.layers.Dense(
            output_dim, activation = 'linear', 
            kernel_initializer = kernel_initializer, name = 'dense_out')
        


    def call(self, l, training = False):
        """ Forward pass function.

        Args:
            l: NN input.
            training: True during training (defaults to False).

        Returns:
            The NN output.
        """
        l = self.dense_in(l)
        l = self.reshape(l) 
        l = self.conv1(l)
        l = self.conv2(l)
        l = self.conv3(l)
        l = self.conv4(l)
        l = self.flatten(l)
        l = self.dense_out(l)
        return l





class ConvTransposeBlock(tf.keras.Model):
    """ Implements a block of 4 transposed convolutions.
    """
    def __init__(
        self, 
        input_image_size,
        output_dim,
        activation = 'gelu',
        kernel_initializer = tf.keras.initializers.Orthogonal
    ):
        """
        
        Args:
            input_image_size (int): the input resolution.
            output_dim (int): the output dimension.
            activation: the nonlinearity (defaults to 'gelu').
            kernel_initializer: for the weights initialization (defaults to 
                                tf.keras.initializers.Orthogonal).
        """

        super(ConvTransposeBlock, self).__init__()

        # Dense layer at input and reshape to get 2D
        self.dense_in = tf.keras.layers.Dense(
            input_image_size**2, activation = activation, 
            kernel_initializer = kernel_initializer, name = 'dense_in')
        self.reshape = tf.keras.layers.Reshape(
            (1, 1, input_image_size**2), name = 'reshape')
        
        # Hidden convolutional layers
        self.conv1_t = tf.keras.layers.Conv2DTranspose(
            64, kernel_size = [5, 5], strides = (2, 2), padding = 'same', 
            activation = activation, kernel_initializer = kernel_initializer, 
            name = 'conv1_t')
        self.conv2_t = tf.keras.layers.Conv2DTranspose(
            32, kernel_size = [5, 5], strides = (2, 2), padding = 'same', 
            activation = activation, kernel_initializer = kernel_initializer, 
            name = 'conv2_t')
        self.conv3_t = tf.keras.layers.Conv2DTranspose(
            16, kernel_size = [5, 5], strides = (2, 2), padding = 'same', 
            activation = activation, kernel_initializer = kernel_initializer, 
            name = 'conv3_t')
        self.conv4_t = tf.keras.layers.Conv2DTranspose(
            1, kernel_size = [5, 5], strides = (1, 1), padding = 'same', 
            activation = activation, kernel_initializer = kernel_initializer, 
            name = 'conv4_t')
        
        # Flatten to get 1D and dense layer at output
        self.flatten = tf.keras.layers.Flatten()
        self.dense_out = tf.keras.layers.Dense(
            output_dim, activation = 'linear', 
            kernel_initializer = kernel_initializer, name = 'dense_out')



    def call(self, l, training = False):
        """ Forward pass function.

        Args:
            l: NN input.
            training: True during training (defaults to False).

        Returns:
            The NN output.
        """
        l = self.dense_in(l)
        l = self.reshape(l)
        l = self.conv1_t(l)
        l = self.conv2_t(l)
        l = self.conv3_t(l)
        l = self.conv4_t(l)
        l = self.flatten(l)
        l = self.dense_out(l)
        return l





class Block(tf.keras.Model):
    """ Implements a generic block of neural networks starting from a list of 
        layers or a list of blocks.
    """

    def __init__(self, blocks : list):
        """
        
        Args:
            blocks (list): list of layers or list of blocks.
        """
        super().__init__()
        self.blocks = blocks
    


    def call(self, l, training = False):
        """ Forward pass function.

        Args:
            l: NN input.
            training: True during training (defaults to False).

        Returns:
            The NN output.
        """
        for i in range(len(self.blocks)):
            l = self.blocks[i](l)
        return l
        
