import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from deepsdf_model import DeepSDFDecoder
from hyperparams import *
from sdf import sdf3
from preprocess import get_mesh_files
from mesh_to_sdf import sample_sdf_near_surface
import trimesh
import pyrender
import mcubes
from sdf import *

model = DeepSDFDecoder(num_shapes, shape_code_dim, hidden_dim, MLP_dropout_rate)
# # Build the model by calling it
x = tf.ones((1, 3))
idx = 0
# # input_arr = [x, idx]
outputs = model([x, idx])
print(outputs)
# model.save("my_model")



#LOAD
loaded_1 = keras.models.load_model("my_model")
# np.testing.assert_allclose(loaded_1(x), outputs)
# print("Original model:", model)
print("Model Loaded with custom objects:", loaded_1)
print(loaded_1([x, idx]))
