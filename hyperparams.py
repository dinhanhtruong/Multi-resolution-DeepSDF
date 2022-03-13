import tensorflow.keras as keras

# ====== GLOBAL HYPERPARAMS ===========
epochs = 1500
batch_sz = 64
hidden_dim = 512
clamping_dist = 0.1
dropout_rate = 0.2
weights_learning_rate = 0.00001 #1e-5 in paper for model weights, 1e-3 for embedding?
shape_code_learning_rate = 0.001
weight_optimizer = keras.optimizers.Adam(weights_learning_rate)
shape_code_optimizer = keras.optimizers.Adam(shape_code_learning_rate)
num_shapes = 5
num_sample_points = 500000 #500000
shape_code_dim = 256 #256 in paper
resample_rate = 10
