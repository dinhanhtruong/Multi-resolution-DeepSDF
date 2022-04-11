import random
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from deepsdf_model import *
from multi_res_hashtable import MultiResolutionHashEncoding
from hyperparams import *
from sdf import sdf3
from preprocess import get_mesh_files
from mesh_to_sdf import sample_sdf_near_surface, mesh_to_sdf
# import mcubes
from sdf import *

def extract_cube(center=(0,0,0)):
    sdf = box(0.3, center=center)
    sdf.save('temp_plane_data/cube/model_normalized.obj' , bounds=((-1,-1,-1), (1, 1, 1)), samples=2**24, sparse=True)

def extract_sphere(center=(0,0,0)):
    sdf = sphere(0.3, center=center)
    sdf.save('temp_plane_data/sphere/model_normalized.obj' , bounds=((-1,-1,-1), (1, 1, 1)), samples=2**24, sparse=True)

def extract_pyramid():
    sdf = box(0.3, center=(0.5, 0.5, 0.5))
    sdf.save('temp_plane_data/pyramid/model_normalized.obj', bounds=((0, 0, 0), (1, 1, 1)), samples=2**24, sparse=True)

def extract_mesh_from_sdf(hashtable, model, filepath, occupancy=False, num_samples=2**25, sparse=False):
    # sdf = trained_sdf(hashtable, shape_code, model, occupancy)
    sdf = trained_sdf(hashtable, model, occupancy)
    print("saving mesh")
    # sdf.save(filepath, bounds=((-1, -1, -1), (1, 1, 1)), samples=num_samples, sparse=sparse)
    sdf.save(filepath+'.stl' , bounds=((0, 0, 0), (1, 1, 1)), samples=num_samples, sparse=sparse)
    print("saved mesh at ", filepath)

@sdf3
def trained_sdf(hashtable, model, occupancy=False):
    '''
    Custom SDF function wrapping the trianed SDF model
    Returns
        f: function representing SDF
    '''
    def f(points):
        if occupancy:
            encoded_positions = hashtable(points)
            out = -np.squeeze(model(encoded_positions).numpy().flatten()) + 0.5
            return out # [N,]  ##================= offset and negate for occupancy only ====
        else:
            # print("f out: ", model.call(points, shape_idx, training=False).numpy().flatten()[:10])
            encoded_positions = hashtable(points)
            out = model(encoded_positions).numpy()
            # print("out: ", out[out<0])
            return np.squeeze(model(encoded_positions).numpy().flatten()) # [N,] 
    return f


# extract_cube()
# extract_sphere()