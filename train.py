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


# ====== TRAINING STEP FOR SINGLE IMAGE ===============
def train_step(shape_idx, positions, sdf_true):
    """Trains the model for a SINGLE shape using the positions given as queries to the SDF
    
    :param shape_idx: 
    :param positions: batch of query positions at which the SDF is queried for training
    :param sdf_true: true SDF values of the shape at positions
    :return: None
    """
    with tf.GradientTape() as tape:
        sdf_pred = model([positions, shape_idx], training=True)
        loss = model.loss(sdf_pred, sdf_true, clamping_dist)

    print("loss: ", loss)
    # train model params and latent codes jointly
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

# ====== MAIN TRAINING LOOP ===============
def train(data_dir):
    path = 'training_checkpoints2'
    # get lexicographically ordered filepaths
    shape_filepaths = get_mesh_files(data_dir)
    print(shape_filepaths)
    for epoch in range(epochs):
        print("================ epoch: ", epoch)
        # iterate over shapes
        for shape_idx, filepath in shape_filepaths.items():
            # TODO: loop over all shapes, convert to SDF, check for bad meshes, enumerate & batch
            mesh = trimesh.load(filepath)
            # convert to sdf
            positions, sdf_vals = sample_sdf_near_surface(mesh, num_sample_points)
            visualize_sdf_points(positions, sdf_vals)
            # batch
            dataset_positions = tf.data.Dataset.from_tensor_slices(positions)
            dataset_sdf = tf.data.Dataset.from_tensor_slices(sdf_vals)
            dataset = tf.data.Dataset.zip((dataset_positions, dataset_sdf)).shuffle(buffer_size=num_sample_points).batch(batch_sz)
            # TODO: convert to fit()
            for batch_positions, batch_sdf_vals in dataset:
                # TODO: check for bad mash exceptions
                train_step(shape_idx, batch_positions, batch_sdf_vals)
            # extract_mesh_from_sdf(shape_idx, model)
            break # train for single shape atm

        # save model every epoch
        print("saving")
        model.save(path)
        print("saved to: ", path)

    extract_mesh_from_sdf(0, model)

def visualize_sdf_points(points, sdf_vals):
    colors = np.zeros(points.shape)
    colors[sdf_vals < 0, 2] = 1
    colors[sdf_vals > 0, 0] = 1
    cloud = pyrender.Mesh.from_points(points, colors=colors)
    scene = pyrender.Scene()
    scene.add(cloud)
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)

def extract_mesh_from_sdf(shape_idx, model):
    sdf = trained_sdf(shape_idx, model)
    print("saving mesh")
    sdf.save('output/out3.stl', bounds=((-1, -1, -1), (1, 1, 1)))
    print("saved mesh")


@sdf3
def trained_sdf(shape_idx, model):
    '''
    Custom SDF function wrapping the trianed SDF model
    Returns
        f: function representing SDF
    '''
    def f(points):
        # print("f out: ", model.call(points, shape_idx, training=False).numpy().flatten()[:10])
        return np.squeeze(model([points, shape_idx]).numpy().flatten()) # [N,]
    return f


if __name__ == "__main__":
    model = DeepSDFDecoder(num_shapes, shape_code_dim, hidden_dim, dropout_rate)
    train("temp_plane_data")