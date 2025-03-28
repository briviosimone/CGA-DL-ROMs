################################################################################
# CONTINUOUS GEOMETRY-AWARE DL-ROM
# Hyper-elasticity equation (Rivlin-Saunders material) -> unit-cell problem
# We refer to https://arxiv.org/pdf/2207.05209.pdf.
#
# Authors:  Simone Brivio, Stefania Fresca, Andrea Manzoni
# Date:     September 2023
################################################################################

import numpy as np
import sys
import random
import matplotlib.pyplot as plt

sys.path.insert(0, '../')

import process
import models
from utils import TestCase, savefig_predictions
from nns import *
from layers import *
from architectures import *

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
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/simone/anaconda3/envs/cga'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


################################################################################
# CONFIGURATION: Set problem configuration
################################################################################

# Set random seed
seed = 1111
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Set save folder
save_folder = '../results/ciao-elasticity/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Create filenames dictionary
filenames = dict()
filenames['S'] = '../data/elasticity/S.npy'
filenames['X'] = '../data/elasticity/X.npy'
filenames['g'] = '../data/elasticity/G.npy'

# Create config dictionary
config = dict()
config['n_epochs'] = 3000
config['lr'] = 3e-4
config['batch_size'] = 8#16
config['d'] = 2
config['N'] = 30#256
config['n'] = 20
config['n_mu'] = 0
config['n_g'] = 42
config['c'] = 1
config['N_train'] = 1000
config['N_test'] = 200
config['continuous'] = True
config['normalize'] = True
config['test_case'] = 'elasticity'



################################################################################
# CONFIGURATION: Define test case
################################################################################

class Elasticity(TestCase): 

    def __init__(self, config, save_folder, filenames):
        super().__init__(config, save_folder, filenames)
        # Postprocessing parameters
        self.idx_plot = 61

    
    def postprocess(self, *args):
        model, test_data, pred_test = args
        X = test_data['X']
        target_test = test_data['S']
        x = X[self.idx_plot,:,0]
        y = X[self.idx_plot,:,1]
        true = target_test[self.idx_plot]
        pred = pred_test[self.idx_plot]
        err = np.abs(true - pred)
        lims = dict(cmap='jet')
        _, axs = plt.subplots(1,3, figsize = (13,5.5))
        im_true = axs[0].scatter(x,y, 38, true, edgecolor='w', lw=0.1, **lims)
        im_pred = axs[1].scatter(x,y, 38, pred, edgecolor='w', lw=0.1, **lims) 
        im_err = axs[2].scatter(x,y, 38, err, edgecolor='w', lw=0.1, **lims)
        cbar_true = plt.colorbar(
            im_true, ax = axs[0], orientation='horizontal', pad=0.02
        )
        cbar_pred = plt.colorbar(
            im_pred, ax = axs[1], orientation='horizontal', pad=0.02
        )
        cbar_err = plt.colorbar(
            im_err, ax = axs[2], orientation='horizontal', pad=0.02
        )
        for cbar in (cbar_true, cbar_pred, cbar_err):
            cbar.ax.tick_params(labelsize = 14)
        #cbar_err.formatter.set_powerlimits((0, 0))
        #cbar_err.formatter.set_useMathText(True)
        for i in range(3):
            axs[i].axis('equal')
            axs[i].spines[['left', 'right', 'top', 'bottom']].set_visible(False)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
        axs[0].set_title('Exact', fontsize = 19, pad = -5)
        axs[1].set_title('Predicted', fontsize = 19, pad = -5)
        axs[2].set_title('Absolute error', fontsize = 19, pad = -5)
        plt.tight_layout()
        plt.subplots_adjust(wspace = 0.23)
        savefig_predictions(model, self.save_folder)


################################################################################
# CONFIGURATION: Construct architectures
################################################################################

# Construct neural networks blocks
reduced_network = DenseNetwork(
    width = 50, 
    depth = 5, 
    output_dim = config['n']
)
encoder = DenseNetwork(
    width = 150, 
    depth = 5, 
    output_dim = config['n']
)
decoder = Block(
    [
        DenseNetwork(
            width = 150, 
            depth = 5, 
            output_dim = config['N']
        ),
        tf.keras.layers.Reshape(
                target_shape = (config['N'], config['c'])
        )
    ]
)
ffnn_global_basis = Block(
    [
        DenseNetwork(
            width = 180, 
            depth = 11, 
            output_dim = config['N'] * config['c'], 
            residual_connections = True
        ),
        tf.keras.layers.Reshape(
                target_shape = (-1, config['N'], config['c'])
        )
    ]
)

def feat_eng_x(x):
    feature_matrix = tf.repeat(
        np.pi * tf.pow(2., tf.cast(tf.range(10), tf.float32))[:,None], 
        axis = 1, 
        repeats = config['d']
    )
    x_f = tf.einsum('bid,kd->bik', x, feature_matrix)
    center = tf.convert_to_tensor(np.array([0.5,0.5]).astype('float32'))
    angle = tf.atan2(
        x[:,:,1] - center[0], x[:,:,0] - center[0])[:,:,None]
    radius = tf.linalg.norm(x - center, axis = 2)[:,:,None]
    x = tf.concat((x, angle, radius), axis = 2)
    x = tf.concat((x, tf.sin(x_f), tf.cos(x_f)), axis = 2)
    return x

global_basis = CGAbasis(
    ffnn = ffnn_global_basis,
    c = config['c'],
    feature_engineering = feat_eng_x
)

architectures_info = dict()
architectures_info['reduced_network'] = reduced_network
architectures_info['encoder'] = encoder
architectures_info['decoder'] = decoder
architectures_info['global_basis'] = global_basis



################################################################################
# MAIN: Perform main computations (training and testing)
################################################################################

# Get option from command line
train_load_test = sys.argv[1]

# Test case instantiation
test_case = Elasticity(
    config = config,
    save_folder = save_folder,
    filenames = filenames
)

# Generate data
data_processor = process.DataProcessor(test_case)
data_processor.loaddata()
data_processor.preprocess()

# Create architectures
model_train, model_inference = create_architectures(
    architectures_info = architectures_info,
    data_processor = data_processor
)

# Instantiate model
model = models.DLROM(
    data_processor = data_processor,
    model_train = model_train,
    model_inference = model_inference
)

# Train, load and train or load model
model.compile()
if train_load_test == 'train':
    model.fit()
elif train_load_test == 'load':
    model.load()
    model.fit()
elif train_load_test == 'test':
    model.load()
else:
    raise('Choose between {train, test, load}')

# Test accuracy
train_error, pred_train, _ = model.compute_error(
    data_processor.train_data
)
val_error, pred_val, _ = model.compute_error(
    data_processor.val_data
)
test_error, pred_test, elapsed_time_per_instance = model.compute_error(
    data_processor.test_data
)
print('\nTESTING PREDICTION ACCURACY')
print('Train error      = ' + str(train_error))
print('Validation error = ' + str(val_error))
print('Test error       = ' + str(test_error))

# Display other info
parameters_count = np.sum(
    [elem.count_params() for elem in (reduced_network, decoder, global_basis)]
)
print('\nOTHER INFO')
print('# NN parameters           = ' + str(parameters_count))
print('Elapsed time per instance = ' + str(elapsed_time_per_instance))

################################################################################
# POSTPROCESSING: Post process the solutions
################################################################################

# Postprocess
data_processor.test_case.postprocess(
    model, 
    data_processor.test_data, 
    pred_test
)

def plot_basis(global_basis):
    idxs_basis = (None,3,5,10)
    idxs_plot = (0,3)
    xy_plot = data_processor.train_data['X'].astype('float32')[1][None,:]
    def plot_basis_series(*args):
        _, axs = plt.subplots(
            2, 
            len(idxs_basis), 
            figsize = (len(idxs_basis) * 3, len(idxs_plot) * 3.1)
        )
        is_reference_plot = (len(args) > 0)
        for idx_axis_x, idx_plot in enumerate(idxs_plot):
            xy_input = data_processor.train_data['X'].astype(
                'float32')[idx_plot][None,:]
            g_input = data_processor.train_data['g'].astype(
                'float32')[idx_plot][None,:]
            basis = global_basis(xy_input, g_input)[0,:,:,0]
            if is_reference_plot:
                xy_plot, = args
            else:
                xy_plot = xy_input
            x_plot = xy_plot[0,:,0]
            y_plot = xy_plot[0,:,1]
            print(g_input)
            for idx_axis_y, idx_basis in enumerate(idxs_basis):
                cmap = 'coolwarm' if idx_axis_y == 0 else 'rainbow'
                to_plot = data_processor.train_data['S'][idx_plot] \
                    if idx_axis_y == 0 else basis[:,idx_basis] 
                im = axs[idx_axis_x,idx_axis_y].scatter(
                    x_plot, 
                    y_plot, 
                    c = to_plot, 
                    s = 12, 
                    cmap = cmap
                ) 
                axs[idx_axis_x,idx_axis_y].spines[
                    ['left', 'right', 'top', 'bottom']].set_visible(False)
                axs[idx_axis_x,idx_axis_y].set_xticks([])
                axs[idx_axis_x,idx_axis_y].set_yticks([])
                axs[idx_axis_x,idx_axis_y].axis('equal')
                if idx_axis_x == 0:
                    if idx_axis_y == 0:
                        if is_reference_plot:
                            to_write = r'$\mathcal{Z}^{-1}_{{\bf{\xi}}}(u({\bf{\xi}}))$'
                        else:
                            to_write = r'$u({\bf{\xi}})$'
                    else:
                        if is_reference_plot:
                            to_write = r'$\mathcal{Z}^{-1}_{{\bf{\xi}}}(\hat{v}_{%d}({\bf{\xi}}))$' % (idx_basis + 1)
                        else:
                            to_write = r'$\hat{v}_{%d}({\bf{\xi}})$' % (idx_basis + 1)
                    axs[idx_axis_x,idx_axis_y].set_title(
                        to_write, fontsize = 18)
                if idx_axis_y == 0:
                    axs[idx_axis_x,idx_axis_y].set_ylabel(
                        r'${\bf{\xi}}_{%d}$' % (idx_axis_x + 1),  
                        fontsize = 18,
                        labelpad = 5
                    )
        plt.tight_layout()
        if is_reference_plot:
            plt.savefig(os.path.join(save_folder, 'basis_functions_ref.png'))
        else:
            plt.savefig(os.path.join(save_folder, 'basis_functions.png'))

    plot_basis_series()
    plot_basis_series(xy_plot)

plt.style.use('dark_background')
plot_basis(global_basis)