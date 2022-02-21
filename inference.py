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


# load model
model = keras.models.load_model("training_checkpoints3_cont")
model.summary()
# try marching cubes
print("extracting")

extract_mesh_from_sdf(0, model, 'output/401_epochs.stl')