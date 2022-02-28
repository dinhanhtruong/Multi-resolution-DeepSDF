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
@tf.function
def train_step(shape_idx, positions, sdf_true):
    """Trains the model for a SINGLE shape using the positions given as queries to the SDF
    
    :param shape_idx: 
    :param positions: batch of query positions at which the SDF is queried for training
    :param sdf_true: true SDF values of the shape at positions
    :return: None
    """
    with tf.GradientTape() as tape:
        sdf_pred = model([positions, shape_idx], training=True)
        # print("model: ", sdf_pred.numpy()[:10])
        # print("actual: ", sdf_true[:10])
        loss = model.loss(sdf_pred, sdf_true, clamping_dist)
        # print("loss: ", loss)

    # print("loss: ", loss)
    # train model params and latent codes jointly
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# ====== MAIN TRAINING LOOP ===============
def train(data_dir, path):
    

    # get lexicographically ordered filepaths
    shape_filepaths = get_mesh_files(data_dir)
    print(shape_filepaths)

    # TODO: loop over all shapes, convert to SDF, check for bad meshes, enumerate & batch
    for shape_idx, filepath in shape_filepaths.items():
        # TEMP: TRAIN FOR EASY PLANE at idx 1
        if shape_idx != 1:
            continue
        shape_idx = 0
        # iterate over shapes
        # convert to sdf
        mesh = trimesh.load(filepath)
        # TODO: check for bad mash exceptions
        positions, sdf_vals = sample_sdf_near_surface(mesh, num_sample_points)
        # convert to occupancy (1 if inside, 0 outside shape)
        # occupancy_vals = np.where(sdf_vals < 0, 1.0, 0.0)
        # visualize_sdf_points(positions, occupancy_vals)
        for epoch in range(epochs):
            losses = []
            print("================ epoch: ", epoch)
            # batch
            dataset_positions = tf.data.Dataset.from_tensor_slices(positions)
            dataset_sdf = tf.data.Dataset.from_tensor_slices(sdf_vals)
            dataset = tf.data.Dataset.zip((dataset_positions, dataset_sdf)).shuffle(buffer_size=num_sample_points).batch(batch_sz, drop_remainder=True)
            for batch_positions, batch_sdf_vals in dataset:
                losses.append(train_step(shape_idx, batch_positions, batch_sdf_vals).numpy())
            # extract_mesh_from_sdf(shape_idx, model)
            avg_loss = np.mean(losses)
            print("epoch loss: ", avg_loss)

            if epoch % 6 == 5:
                # save model every epoch
                print("saving")
                model.save(path)
                print("saved to: ", path)
        break # train for single shape atm

    # extract_mesh_from_sdf(0, model, 'output/out4.stl')

def visualize_sdf_points(points, sdf_vals):
    colors = np.zeros(points.shape)
    colors[sdf_vals < 0, 2] = 1
    colors[sdf_vals > 0, 0] = 1
    cloud = pyrender.Mesh.from_points(points, colors=colors)
    scene = pyrender.Scene()
    scene.add(cloud)
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)

def extract_mesh_from_sdf(shape_idx, model, filepath, occupancy=False, num_samples=2**25):
    sdf = trained_sdf(shape_idx, model, occupancy)
    print("saving mesh")
    sdf.save(filepath, bounds=((-1, -1, -1), (1, 1, 1)), samples=num_samples)
    print("saved mesh at ", filepath)

@sdf3
def trained_sdf(shape_idx, model, occupancy=False):
    '''
    Custom SDF function wrapping the trianed SDF model
    Returns
        f: function representing SDF
    '''
    def f(points):
        if occupancy:
            return -np.squeeze(model([points, shape_idx]).numpy().flatten()) + 0.5 # [N,]  ##================= offset and negate for occupancy only ====
        else:
            # print("f out: ", model.call(points, shape_idx, training=False).numpy().flatten()[:10])
            return np.squeeze(model([points, shape_idx]).numpy().flatten()) # [N,] 
    return f

def random_ball(num_points, dimension, radius=1):
    # First generate random directions by normalizing the length of a
    # vector of random-normal values (these distribute evenly on ball).
    random_directions = np.random.normal(size=(dimension,num_points))
    random_directions /= np.linalg.norm(random_directions, axis=0)
    # Second generate a random radius with probability proportional to
    # the surface area of a ball with a given radius.
    random_radii = np.random.random(num_points) ** (1/dimension)
    # Return the list of random (direction & length) points.
    return radius * (random_directions * random_radii).T

if __name__ == "__main__":
    # model = DeepSDFDecoder(num_shapes, shape_code_dim, hidden_dim, dropout_rate)
    model = keras.models.load_model('training_checkpoints_scaled_output')
    save_path = 'training_checkpoints_scaled_output'
    train("temp_plane_data", save_path)