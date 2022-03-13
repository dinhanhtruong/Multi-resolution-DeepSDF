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
model_path = 'trained_models/' + 'multishape_5shapes_occupancy_1e-3emb_model_499epochs'
shape_code_path = 'trained_models/' + 'multishape_5shapes_occupancy_1e-3emb_emb_499epochs'

save_dir = 'output/' + 'multishape/5shapes_occupancy_2000epochs_shape4_0475ls' + '.stl'  # MAKE NEW DIR

# load model
model = keras.models.load_model(model_path)
shape_codes = keras.models.load_model(shape_code_path)
model.summary()


print("extracting")

# extract_mesh_from_sdf(4, model, save_dir, occupancy=True, num_samples=2**27, sparse=False) #2**25 HI, 2**27 very high, 2**22 default



for shape_idx in range(num_shapes):
    save_dir = 'output/' + 'multishape/5shapes_occupancy_1e-3emb_500epochs_shape' + str(shape_idx) + '.stl'  # MAKE NEW DIR
    shape_code = shape_codes(shape_idx)
    print(shape_code[:20])
    # extract_mesh_from_sdf(shape_code, model, save_dir, occupancy=True, num_samples=2**27, sparse=False) #2**25 HI, 2**27 very high, 2**22 default