import random
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from deepsdf_model import *
from hyperparams import *
from sdf import sdf3
from preprocess import get_mesh_files
from mesh_to_sdf import sample_sdf_near_surface
import trimesh
import pyrender
# import mcubes
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
    with tf.GradientTape(persistent=True) as tape:
        shape_code = shape_codes(shape_idx)
        sdf_pred = model([positions, shape_code], training=True)
        # print("model: ", sdf_pred.numpy()[:10])
        # print("actual: ", sdf_true[:10])
        loss = model.loss(sdf_pred, sdf_true)
    # print("loss: ", loss)

    # train model params and latent codes jointly
    weight_grads = tape.gradient(loss, model.trainable_variables)
    weight_optimizer.apply_gradients(zip(weight_grads, model.trainable_variables))
    shape_code_grads = tape.gradient(loss, shape_codes.trainable_variables)
    shape_code_optimizer.apply_gradients(zip(shape_code_grads, shape_codes.trainable_variables))
    return loss

# ====== MAIN TRAINING LOOP ===============
def train(data_dir, model_save_path, shape_code_save_path):
    

    # get lexicographically ordered filepaths
    shape_filepaths = get_mesh_files(data_dir)
    print(shape_filepaths)

    # TODO: loop over all shapes, convert to SDF, check for bad meshes, enumerate & batch
    for epoch in range(epochs):
        print("======= epoch: ", epoch*3)

        # pick random shape 
        shape_idx = random.randint(0,num_shapes-1)
        print("shape idx: ", shape_idx)
        filepath = shape_filepaths[shape_idx]
        mesh = trimesh.load(filepath)

        # convert to sdf
        # TODO: check for bad mash exceptions? exception can be thrown if < 1.5% of uniformly sampled points have negative SDFs (ie occupied)
        positions, sdf_vals = sample_sdf_near_surface(mesh, num_sample_points)
        # convert to occupancy (1 if inside, 0 outside shape)
        occupancy_vals = np.where(sdf_vals < 0, 1.0, 0.0)
        # visualize_sdf_points(positions, sdf_vals)
        dataset_positions = tf.data.Dataset.from_tensor_slices(positions)
        dataset_sdf = tf.data.Dataset.from_tensor_slices(occupancy_vals)
        dataset = tf.data.Dataset.zip((dataset_positions, dataset_sdf)).shuffle(buffer_size=num_sample_points).batch(batch_sz, drop_remainder=True)

        # train for 3 epochs
        for _ in range(3):
            losses = []
            # batch
            for batch_positions, batch_occupancy_vals in dataset:
                losses.append(train_step(shape_idx, batch_positions, batch_occupancy_vals).numpy())
            print("epoch loss: ", np.mean(losses))

        if epoch % 14 == 13:
            # save model every few epochs
            print("saving...")
            model.save(model_save_path)
            shape_codes.save(shape_code_save_path)
            print("saved to: ", model_save_path)
        if epoch % 500 == 499:
            print("saving checkpoint at ", str(epoch), " epochs")
            model.save(model_save_path+"_"+ str(epoch)+"epochs")
            shape_codes.save(shape_code_save_path+"_"+ str(epoch)+"epochs")
            print("saved to: ", model_save_path)

    # extract_mesh_from_sdf(0, model, 'output/out4.stl')

def visualize_sdf_points(points, sdf_vals):
    colors = np.zeros(points.shape)
    colors[sdf_vals < 0, 2] = 1
    colors[sdf_vals > 0, 0] = 1
    cloud = pyrender.Mesh.from_points(points, colors=colors)
    scene = pyrender.Scene()
    scene.add(cloud)
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)

def extract_mesh_from_sdf(shape_code, model, filepath, occupancy=False, num_samples=2**25, sparse=True):
    sdf = trained_sdf(shape_code, model, occupancy)
    print("saving mesh")
    sdf.save(filepath, bounds=((-1, -1, -1), (1, 1, 1)), samples=num_samples, sparse=sparse)
    print("saved mesh at ", filepath)

@sdf3
def trained_sdf(shape_code, model, occupancy=False):
    '''
    Custom SDF function wrapping the trianed SDF model
    Returns
        f: function representing SDF
    '''
    def f(points):
        if occupancy:
            return -np.squeeze(model([points, shape_code]).numpy().flatten()) + 0.5 # [N,]  ##================= offset and negate for occupancy only ====
        else:
            # print("f out: ", model.call(points, shape_idx, training=False).numpy().flatten()[:10])
            return np.squeeze(model([points, shape_code]).numpy().flatten()) # [N,] 
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
    data_dir = "temp_plane_data"
    trained_model_dir = 'multishape_5shapes_gaussian'
    save_dir = 'multishape_5shapes_gaussian'

    trained_model_path = 'trained_models/' + trained_model_dir + '_model'
    trained_shape_code_path = 'trained_models/' + trained_model_dir + '_emb'
    model_save_path = 'trained_models/' + save_dir + '_model'
    shape_code_save_path = 'trained_models/' + save_dir + '_emb'

    # model = DeepSDFDecoder(num_shapes, shape_code_dim, hidden_dim, dropout_rate)
    # shape_codes = ShapeCodeEmbedding(num_shapes, shape_code_dim)
    model = keras.models.load_model(model_save_path)
    shape_codes = keras.models.load_model(shape_code_save_path)

    train(data_dir, model_save_path, shape_code_save_path)
    