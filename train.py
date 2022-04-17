from sdf_decoder import SDFDecoder
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from sdf_decoder import *
from multi_res_hashtable import MultiResolutionHashEncoding
from hyperparams import *
from sdf import sdf3
from preprocess import get_mesh_files, SDFSampleDataset
from mesh_to_sdf import sample_sdf_near_surface
import trimesh
import pyrender
from sdf import *
import os
# for DeepSDF-style sampling via SSH
# os.environ['DISPLAY'] = ':1'

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# ====== MAIN TRAINING LOOP ===============
def train(dataloader, model, hashtable_enc, hashtable_save_path, model_save_path):
    params = list(model.parameters()) + list(hashtable_enc.parameters())
    optimizer= torch.optim.Adam(params, learning_rate, (beta_1,beta_2), epsilon, weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    size = len(dataloader.dataset)
    model.train()
    hashtable_enc.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        for batch, data in enumerate(dataloader):
            shape_idx, positions, sdf_vals = data
            positions = positions.to(device)
            sdf_vals = sdf_vals.to(device)
            
            positions = torch.squeeze(positions, dim=0) 
            sdf_vals = torch.squeeze(sdf_vals, dim=0) 
            # positions, sdf_vals = positions.to(device), sdf_vals.to(device)

            # TODO: use list of hashtable enc to index

            encoded_positions = hashtable_enc(positions)
            sdf_pred = model(encoded_positions)
            loss = loss_fn(sdf_pred, sdf_vals)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 50 == 0:
                loss, current = loss.item(), batch
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            # if batch % 1000 == 0:
        # save model/checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'hashtable_state_dict': hashtable_enc.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, model_save_path + '_epoch{}.pt'.format(epoch, batch))
        print("saved " + model_save_path + '_epoch{}.pt'.format(epoch))

        # # update scheduler
        # scheduler.step()

    print("Done!")

def visualize_sdf_points(points, sdf_vals):
    colors = np.zeros(points.shape)
    colors[sdf_vals < 0, 2] = 1
    colors[sdf_vals > 0, 0] = 1
    cloud = pyrender.Mesh.from_points(points, colors=colors)
    scene = pyrender.Scene()
    scene.add(cloud)
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)


if __name__ == "__main__":
    trained_model_dir = 'hashtable/torch_base/plane2_randsampling_512b_recenter'
    new_model_save_dir = 'hashtable/torch_base/plane2_randsampling_512b_recenter'

    # prepare data
    data_dir = "temp_plane_data"
    sdf_dirs = ['temp_plane_data/plane2']
    dataset = SDFSampleDataset(sdf_dirs, batch_sz)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True) #set n=1 because already batched in dataset

    # for loading trained models
    trained_model_path = 'trained_models/' + trained_model_dir + '_model'
    trained_hashtable_path = 'trained_models/' + trained_model_dir + '_table'
    # for training models from scratch
    model_save_path = 'trained_models/' + new_model_save_dir + '_model'
    hashtable_save_path = 'trained_models/' + new_model_save_dir + '_table'


    # initialize models
    hashtable_enc = MultiResolutionHashEncoding(table_sz, max_resolution)
    hashtable_enc.to(device)
    model = SDFDecoder(encoded_position_dim, hidden_dim)
    model.to(device)


    # load existing models
    # hashtable_enc = keras.models.load_model(hashtable_save_path)
    # model = keras.models.load_model(model_save_path)
    # shape_codes = keras.models.load_model(shape_code_save_path)

    train(dataloader, model, hashtable_enc, hashtable_save_path, model_save_path)
    