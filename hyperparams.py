import tensorflow.keras as keras
from matplotlib import pyplot
import numpy as np
from hyperparams import *
from deepsdf_model import DeepSDFDecoder

# ====== GLOBAL HYPERPARAMS ===========
epochs = 1
batch_sz = 64
hidden_dim = 512
clamping_dist = 0.1
dropout_rate = 0.2
learning_rate = 0.00001
optimizer = keras.optimizers.Adam(learning_rate)
num_shapes = 1
num_sample_points = 200 # 25000 #500000
shape_code_dim = 128 #256 in paper