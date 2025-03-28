################################################################################
# CONTINUOUS GEOMETRY-AWARE DL-ROM
# Implementation of neural network models
#
# Authors:  Simone Brivio, Stefania Fresca, Andrea Manzoni
# Date:     July 2023
################################################################################

import time
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

from utils import plot_history, save_history, relative_error


# SET RANDOM SEED
seed = 1111
np.random.seed(seed)
tf.random.set_seed(seed)



class DLROM:
    """ Implements the DLROM model training and inference phases
    """


    def __init__(
        self, 
        data_processor, 
        model_train : tf.keras.Model, 
        model_inference : tf.keras.Model, 
        model_name : str = ''
    ):
        """

        Args:
            data_processor: useful for per-processing.
            model_train (tf.keras.Model): training model (with encoder).
            model_inference (tf.keras.Model): inference model (without encoder).
            model_name (str): name of the model (defaults to '')
        """

        # Extract config parameters
        config = data_processor.config
        self.d = config['d']
        self.c = config['c']
        self.N = config['N']
        self.n = config['n']
        self.n_mu = config['n_mu']
        self.n_g = config['n_g']
        self.n_epochs = config['n_epochs']
        self.batch_size = config['batch_size']
        self.test_case_name = config['test_case'] 
        self.lr = config['lr']

        # Define data processor
        self.data_processor = data_processor

        # Store models and model info
        self.model_train = model_train
        self.model_inference = model_inference
        self.model_name = model_name
        
        # For saving and checkpointing
        self.save_folder = data_processor.test_case.save_folder
        checkpoint_folder = self.save_folder + "/dlrom_checkpoints/"
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        curr_filepath = checkpoint_folder + model_name 
        if model_name != '':
            curr_filepath = curr_filepath + '_'
        self.checkpoint_filepath =  curr_filepath + "dlrom_weights.h5"
        self.saved_loss = float('inf')
        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True
        )

        # Print summary
        self.model_train.summary()



    def loss_output(self, y_true, y_pred):
        """ Reconstruction loss computation.

        Args:
            y_true: targets.
            y_pred: predictions.
        
        Return:
            The computed loss.
        """
        max_N_h = y_true.shape[1]
        y_output = y_pred[:,:max_N_h]
        mask = y_pred[:,max_N_h:]
        N_h = tf.reduce_sum(mask, axis = 1)[:,None]
        reconstruction_error =  (y_output - y_true) * mask 
        squared_error = tf.reduce_sum(reconstruction_error**2, axis = 1)
        return tf.reduce_mean(mask.shape[1] / N_h * squared_error)



    def metric(self, y_true, y_pred):
        """ Relative error metric computation.

        Args:
            y_true: targets.
            y_pred: predictions.
        
        Return:
            The computed metric.
        """
        max_N_h = y_true.shape[1]
        y_output = y_pred[:,:max_N_h]
        mask = y_pred[:,max_N_h:]
        diff = (y_output - y_true) * mask
        return tf.reduce_mean(
            tf.linalg.norm(diff, axis = 1) / \
            tf.linalg.norm(y_true * mask, axis = 1)
        )



    def compile(self):
        """ Compile function.
        """

        self.model_train.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr),
            loss = [self.loss_output],
            metrics = [self.metric],
            jit_compile = True
        )



    def fit(self):
        """ Training function.

        Return:
            The training history.
        """

        # Extract data
        x_train = self.data_processor.train_data
        y_train = self.data_processor.train_data['S']
        x_val = self.data_processor.val_data
        y_val = self.data_processor.val_data['S']

        #Training
        history = self.model_train.fit(
            x_train, 
            y_train, 
            batch_size = self.batch_size, 
            epochs = self.n_epochs, 
            validation_data = (x_val, y_val),
            callbacks = [self.model_checkpoint_callback]
        )
        
        # Training post-processing
        plot_history(self, history, self.save_folder)
        save_history(self, history, self.save_folder)

        return history



    def load(self):
        """ To load model weights.
        """
        self.model_train.load_weights(self.checkpoint_filepath)
    
    

    def compute_error(self, data, L2 : bool = False):
        """ Inference error computation.
        
        Args:
            data: the labeled data.
            L2 (bool): if True, we use the L^2(P) norm (defaults to False).

        Return:
            The test error, the output solutions, the elapsed time per instance.
        """

        # Load best model
        self.load()

        # Extract input data and targets
        mask = data['mask']
        x_test = {
            key : data[key] for key in (set(data.keys()) - set(('mask', 'S')))
        }
        y_true = data['S']

        # Forward pass and elapsed time
        t0 = time.time()
        y_output = self.model_inference.predict(x_test)
        t1 = time.time()
        elapsed_time = t1 - t0
        elapsed_time_per_instance = elapsed_time / y_true.shape[0]

        # Relative error computation
        test_err = relative_error(
            np.einsum('ijk,ij->ijk', y_true, mask), 
            np.einsum('ijk,ij->ijk', y_output, mask), 
            L2
        )
        
        return test_err, y_output, elapsed_time_per_instance
    