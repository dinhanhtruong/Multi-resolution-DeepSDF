import tensorflow as tf
import tensorflow.keras as keras
from matplotlib import pyplot
import numpy as np
from deepsdf_model import DeepSDFDecoder
from sdf import sdf3
from train import extract_mesh_from_sdf, visualize_sdf_points
from preprocess import get_mesh_files
from mesh_to_sdf import sample_sdf_near_surface
import trimesh
import pyrender
from sdf import *
from hyperparams import *

# mesh = trimesh.load_mesh('out.stl')
# mesh.show()
hashtable_save_path = 'trained_models/' + 'hashtable/test2/base_4layers_512res_2e14table_table'
shape_code_path = 'trained_models/' + 'hashtable/test2/base_4layers_512res_2e14table_emb'
model_path = 'trained_models/' + 'hashtable/test2/base_4layers_512res_2e14table_model'


save_dir = 'output/' + 'hashtable/base/base_4layers_512res_2e14table_450epochs' + '.stl'  # MAKE NEW DIR

# load model
hashtable_enc = keras.models.load_model(hashtable_save_path)
model = keras.models.load_model(model_path)
shape_codes = keras.models.load_model(shape_code_path)
hashtable_enc.summary()
model.summary()
shape_codes.summary()


print("extracting")
shape_idx = 1
shape_code = shape_codes(shape_idx)
# print(shape_code[:20])
extract_mesh_from_sdf(hashtable_enc, shape_code, model, save_dir, occupancy=True, num_samples=2**25, sparse=True) #2**25 HI, 2**27 very high, 2**22 default


# for shape_idx in range(num_shapes):
#     save_dir = 'output/' + 'multishape/gaussian/5shapes_gaussian_model_2000epochs' + '_shape'+str(shape_idx) + '.stl'  # MAKE NEW DIR
#     shape_code = shape_codes(shape_idx)
#     # print(shape_code[:20])
#     extract_mesh_from_sdf(shape_code, model, save_dir, occupancy=True, num_samples=2**27, sparse=False) #2**25 HI, 2**27 very high, 2**22 default