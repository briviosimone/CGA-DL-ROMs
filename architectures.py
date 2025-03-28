################################################################################
# CONTINUOUS GEOMETRY-AWARE DL-ROM
# Implementation of full neural network architectures_info
#
# Authors:  Simone Brivio, Stefania Fresca, Andrea Manzoni
# Date:     July 2023
################################################################################


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

from layers import CGAProjLayer, CGALiftLayer, PODProjLayer, PODLiftLayer



# SET RANDOM SEED
seed = 1111
tf.random.set_seed(seed)




def create_architectures(architectures_info : dict, data_processor):
    """
    Constructs the DLROM full architecture

    Args:
        architectures_info (dict): dict containing architecture components.
        data_processor: provides in-place preprocessing utilities.

    Returns:
        The training and the inference model
    """

    config = data_processor.config
    
    # Extract network components
    encoder = architectures_info['encoder']
    reduced_network = architectures_info['reduced_network']
    decoder = architectures_info['decoder']
    normalizer = data_processor.normalizer
    if data_processor.test_case.config['continuous']:
        global_basis = architectures_info['global_basis']
    
    # Read inputs and eventually normalize them
    input_g = tf.keras.layers.Input((config['n_g']), name = 'g')
    input_loc = tf.keras.layers.Input((None, config['d']), name = 'X')
    input_eval = tf.keras.layers.Input((None, config['c']), name = 'S')
    if normalizer is not None:
        data_in = dict()
        data_in['g'] = input_g
        data_in['X'] = input_loc
        data_in['S'] = input_eval
    n_mu = config.get('n_mu')
    if n_mu is None:
        n_mu = 0
    if n_mu > 0:
        input_mu = tf.keras.layers.Input((n_mu), name = 'mu')
        if normalizer is not None:
            data_in['mu'] = input_mu
            data_out = normalizer.forward(data_in)
            mu, g, loc, eval = data_out['mu'], data_out['g'], \
                data_out['X'], data_out['S']
        else:
            mu, g, loc, eval = input_mu, input_g, input_loc, input_eval
        concat_mu_g = tf.concat((mu, g), axis = 1)
    else:
        if normalizer is not None:
            data_out = normalizer.forward(data_in)
            g, loc, eval = data_out['g'], data_out['X'], data_out['S']
        else:
            g, loc, eval = input_g, input_loc, input_eval
        concat_mu_g = g

    # Modify mask to adapt to channels number
    input_mask = tf.keras.layers.Input((None,1), name = 'mask')
    mask = tf.tile(input_mask, multiples = (1,1,config['c']))

    # ENCODER NETWORK
    if data_processor.test_case.config['continuous']:
        projected_input = CGAProjLayer(global_basis)(loc, eval * mask, g)
    else:
        discr_proj = PODProjLayer(data_processor.train_data['S'], config['N'])
        projected_input = discr_proj(eval)
    output_enc = encoder(projected_input)

    # REDUCED NETWORK
    output_red = reduced_network(concat_mu_g)
    
    # DECODER NETWORK
    output_dec = decoder(output_red)
    if data_processor.test_case.config['continuous']:
        output = CGALiftLayer(global_basis)(loc, output_dec, g)
    else:
        output = PODLiftLayer(discr_proj.basis)(output_dec)

    # Reverse normalization of the output
    if normalizer is not None:
        output = normalizer.backward(output) 

    # Construction of the training model
    if config['n_mu'] > 0:
        inputs = [input_mu, input_g, input_loc, input_eval, input_mask]
        outputs = tf.concat([output, mask], axis = 1)
    else:
        inputs = [input_g, input_loc, input_eval, input_mask]
        outputs = tf.concat([output, mask], axis = 1)
    model_train = tf.keras.models.Model(
        inputs = inputs,
        outputs = outputs
    )

    # Adding internal loss to the training model
    L_inner = tf.reduce_mean(
        tf.reduce_sum((output_red - output_enc)**2, axis = 1)
    )
    model_train.add_loss(L_inner)

    # Construction of the inference model
    if config['n_mu'] > 0:
        inputs = [input_mu, input_g, input_loc]
        outputs = output
    else:
        inputs = [input_g, input_loc]
        outputs = output
    model_inference = tf.keras.models.Model(
        inputs = inputs,
        outputs = outputs
    )

    return model_train, model_inference

