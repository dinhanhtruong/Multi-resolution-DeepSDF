import tensorflow as tf
import tensorflow.keras as keras
from matplotlib import pyplot
import numpy as np
from deepsdf import DeepSDFDecoder

# ====== GLOBAL HYPERPARAMS ===========
epochs = 1
batch_sz = 64
hidden_dim = 512
latent_dim = 256
dropout_rate = 0.2
learning_rate = 0.001
optimizer = keras.optimizers.Adam(learning_rate = 0.001)
model = DeepSDFDecoder(latent_dim, hidden_dim, dropout_rate)
num_shapes = ?
shape_code_dim = ?

# ====== TRAINING STEP FOR SINGLE IMAGE ===============
def train_step(shape_idx, positions, sdf_true):
    """Trains the model for a SINGLE shape using the positions given as queries to the SDF
    
    :param shape_idx: 
    :param positions: batch of query positions at which the SDF is queried for training
    :param sdf_true: true SDF values of the shape at positions
    :return: None
    """
    with tf.GradientTape() as tape:
        sdf_pred = model.call(positions, shape_idx, training=True)
        loss = model.loss(sdf_pred, sdf_true)
    # train model params and latent codes jointly
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

# ====== MAIN TRAINING LOOP ===============
def train():
    checkpoint = tf.train.Checkpoint(model=model) 
    checkpoint_dir_prefix = "training_checkpoints2/checkpoint"
    for epoch in range(epochs):
        # TODO: pass in shape index based on ShapeNet ordering, query positions (close to surface or rand?), true signed distance at positions
        train_step(_, _, _)

        # save model every epoch
        path = checkpoint.save(checkpoint_dir_prefix)
        print("checkpoint saved at: ", path)


if __name__ == "__main__":
    model = DeepSDFDecoder(latent_dim, num_shapes, shape_code_dim, hidden_dim, dropout_rate)
    train()