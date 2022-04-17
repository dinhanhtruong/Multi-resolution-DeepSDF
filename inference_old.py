import tensorflow as tf
import tensorflow.keras as keras
from matplotlib import pyplot
import numpy as np
from deepsdf_model_old import DeepSDFDecoder
from sdf import sdf3
from train_old import visualize_sdf_points
from preprocess import get_mesh_files
from mesh_to_sdf import sample_sdf_near_surface
import trimesh
from sdf import *
from hyperparams import *
from mcwrapper import extract_mesh_mcubes
from oldmc import extract_mesh_from_sdf

# ========== NOTE: MUST MAKE DIRECTORY FIRST! ===============#
trained_dir =  'hashtable/base14/plane2_sample60w_lowmaxresb138'
# trained_dir =  'hashtable/cube/cube'
epoch_list = [0,1,2,10,20,29]

for num_epochs in epoch_list:

    # mesh = trimesh.load_mesh('out.stl')
    # mesh.show()
    hashtable_save_path = 'trained_models/' + trained_dir + '_table' + '_'+str(num_epochs)+'epochs'
    # shape_code_path = 'trained_models/' + trained_dir + '_emb'
    model_path = 'trained_models/' + trained_dir + '_model' + '_'+str(num_epochs)+'epochs'

    save_dir = 'output/' + trained_dir + '_'+str(num_epochs)+'epochs' # MAKE NEW DIR

    # load model
    hashtable_enc = keras.models.load_model(hashtable_save_path)
    model = keras.models.load_model(model_path)
    # shape_codes = keras.models.load_model(shape_code_path)
    # hashtable_enc.summary()
    # model.summary()
    # shape_codes.summary()


    print("extracting")
    shape_idx = 1
    # shape_code = shape_codes(shape_idx)
    # print(shape_code[:20])

    # extract_mesh_from_sdf(hashtable_enc, model, save_dir, occupancy=False, num_samples=2**26, sparse=False) #2**25 HI, 2**27 very high, 2**22 default
    extract_mesh_mcubes(hashtable_enc, model, save_dir)
    
    print("done")