from matplotlib import pyplot
import numpy as np
import torch
from sdf_decoder import SDFDecoder
from multi_res_hashtable import MultiResolutionHashEncoding
from sdf import sdf3
from train import visualize_sdf_points
from preprocess import get_mesh_files
from mesh_to_sdf import sample_sdf_near_surface
import trimesh
from sdf import *
from hyperparams import *
from mcwrapper import extract_mesh_mcubes
from oldmc import extract_mesh_from_sdf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

trained_dir_name = 'torch_base/plane2_randsampling_512b_recenter'

# ========== NOTE: MUST MAKE DIRECTORY FIRST! ===============#
prefix = 'hashtable/{}'.format(trained_dir_name)
suffixes = [
    '_model_epoch0',
    '_model_epoch1',
    '_model_epoch3',
]


for suffix in suffixes:
    checkpoint_dir = 'trained_models/' + prefix + suffix + '.pt'
    save_dir = 'output/' + prefix + suffix

    # load model
    checkpoint = torch.load(checkpoint_dir)

    hashtable_enc = MultiResolutionHashEncoding(table_sz, max_resolution)
    hashtable_enc.load_state_dict(checkpoint['hashtable_state_dict'])
    hashtable_enc.eval()
    model = SDFDecoder(encoded_position_dim, hidden_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # hashtable_enc.to(device)
    # model.to(device)

    

    print("extracting")
    shape_idx = 1
    # CHECK MC BOUNDS
    extract_mesh_from_sdf(hashtable_enc, model, save_dir, occupancy=False, num_samples=2**26, sparse=False) #2**25 HI, 2**27 very high, 2**22 default
    # extract_mesh_mcubes(hashtable_enc, model, save_dir)
    
    print("done")