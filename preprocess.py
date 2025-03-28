################################################################################
# CONTINUOUS GEOMETRY-AWARE DL-ROM
# Preprocessing
#
# Authors:  Simone Brivio, Stefania Fresca, Andrea Manzoni
# Date:     July 2023
################################################################################

import numpy as np
import os

# Load data
parent_folder = os.path.join('data', 'elasticity', 'Meshes')
S = np.load(os.path.join(parent_folder, 'Random_UnitCell_sigma_10.npy'))
S = S.transpose((1,0))
X = np.load(os.path.join(parent_folder, 'Random_UnitCell_XY_10.npy'))
X = X.transpose((2,0,1))
G = np.load(os.path.join(parent_folder, 'Random_UnitCell_rr_10.npy'))
G = G.transpose((1,0))

# Retain part of the data
S =  np.concatenate((S[:1000],S[-400:]), axis = 0)
X = np.concatenate((X[:1000],X[-400:]), axis = 0)
G =  np.concatenate((G[:1000],G[-400:]), axis = 0)

# Save data
save_dir = os.path.join('data','elasticity')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
np.save(os.path.join(save_dir, 'S.npy'), S)
np.save(os.path.join(save_dir, 'X.npy'), X)
np.save(os.path.join(save_dir, 'G.npy'), G)