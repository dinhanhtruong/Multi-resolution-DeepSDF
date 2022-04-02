import tensorflow.keras as keras

# ====== GLOBAL HYPERPARAMS ===========
epochs = 1000
batch_sz = 64

# preprocess/data
num_sample_points = 100000 #500000

# Shape code embedding
num_shapes = 2
shape_code_dim = 128 #256 in paper

# hashtable 
table_sz = 2**18 # num entries per table, [2*14, 2**24]
max_resolution = 2**18 # max grid length sz [2^9=512, 512*1024=524288]

#MLP
hidden_dim = 64 # 512   must be at least shape_code_dim + encoded_pos_dim
MLP_dropout_rate = 0.2
encoded_position_dim = 16*2 #16 = num hashtable levels (fixed), 2=feature dimension of a table entry (fixed)

#training
weights_learning_rate = 0.00001 #1e-5 in paper for model weights, 1e-3 for embedding?
shape_code_learning_rate = 0.00001
MLP_optimizer = keras.optimizers.Adam(weights_learning_rate)
shape_code_optimizer = keras.optimizers.Adam(shape_code_learning_rate)
hashtable_optimizer = keras.optimizers.Adam(beta_2=0.99, epsilon=10**(-15))

