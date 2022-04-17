import os
import torch
from torch.utils.data import Dataset
from hyperparams import *
import numpy as np

data_root_dir = 'temp_plane_data'

def get_mesh_files(data_root_dir):
    '''
    Builds a dictionary mapping shape indices to filepaths for all .obj files in the 
    given directory (searches through subdirectories as well). Indices are based on the
    lexicographical ordering of the files in the directory.
    
    :return: dict of shape idx->filepath  lexicographical order
    '''
    filepaths = {}
    idx = 0
    for dirpath, dirnames, filenames in os.walk(data_root_dir):
        for file in filenames:
            if file.endswith(".obj"):
                obj_path = os.path.join(dirpath, file)
                filepaths[idx] = obj_path
                idx += 1
                # print(obj_path)

    return filepaths

class SDFSampleDataset(Dataset):
    def __init__(self, sdf_dirs, batch_sz=512):
        '''
        Args:
            sdf_dirs: list of string directories, each containing cloud_points.npz and surface_points.npz
                for samples of a single shape. e.g. ['data/plane1', 'data/plane2', ...]
        '''
        self.dir_list = sdf_dirs
        self.num_files = len(sdf_dirs)
        self.batch_sz = batch_sz
        self.num_sample_points = num_sample_points_cloud + num_sample_points_surface
        

    def __len__(self):
        # return number of full BATCHES across ALL directories
        total_num_samples = (self.num_files*(self.num_sample_points)) 
        return total_num_samples // self.batch_sz

    def __getitem__(self, global_batch_idx):
        '''
        Args:
            global_batch_idx: integer from [0, total number of batches across all shapes]
        Returns:
            shape_idx: idx of the shape associated with the given global_batch_idx
            positions: normalized 3D positions of the samples in the batch in [0,1]
            sdf_vals: sdf values associated with the positions
        '''
        # find shape associated with curr idx
        num_batches_in_one_file = (self.num_sample_points) // batch_sz 
        shape_idx = global_batch_idx // num_batches_in_one_file

        sample_points = np.load(self.dir_list[shape_idx] + '/all_points.npz')
        
        # get 1:1 ratio of on-surface and off-surface sample points 
        # note: last 100k samples are uniform, rest are exp weighted    
        positions = sample_points['points'] 

        # get corresponding SDF values
        distances = sample_points['distances']

        
        local_batch_idx = global_batch_idx % num_batches_in_one_file
        
        return shape_idx, positions[local_batch_idx: local_batch_idx + self.batch_sz], distances[local_batch_idx: local_batch_idx + self.batch_sz]

# class SDFSampleDataset(Dataset):
#     def __init__(self, sdf_dirs, batch_sz=512):
#         '''
#         Args:
#             sdf_dirs: list of string directories, each containing cloud_points.npz and surface_points.npz
#                 for samples of a single shape. e.g. ['data/plane1', 'data/plane2', ...]
#         '''
#         self.dir_list = sdf_dirs
#         self.num_files = len(sdf_dirs)
#         self.batch_sz = batch_sz
#         self.num_sample_points = num_sample_points_cloud + num_sample_points_surface
        

#     def __len__(self):
#         # return number of full BATCHES across ALL directories
#         return self.num_files

#     def __getitem__(self, shape_idx):
#         '''
#         Args:
#             global_batch_idx: integer from [0, total number of batches across all shapes]
#         Returns:
#             shape_idx: idx of the shape associated with the given global_batch_idx
#             positions: normalized 3D positions of the samples in the batch in [0,1]
#             sdf_vals: sdf values associated with the positions
#         '''

#         sample_points_non_surface = np.load(self.dir_list[shape_idx] + '/cloud_points.npz')
#         sample_points_surface = np.load(self.dir_list[shape_idx] + '/surface_points.npz')
        
        

#         # get 1:1 ratio of on-surface and off-surface sample points 
#         # note: last 100k samples are uniform, rest are exp weighted    
#         positions_non_surface = sample_points_non_surface['cloud_points'] 
#         positions_surface = sample_points_surface['surface_points']
        
#         positions = np.vstack((positions_non_surface, positions_surface))
#         positions = 0.5*positions + 0.5

#         # get corresponding SDF values
#         sdf_vals_non_surface = sample_points_non_surface['distances']
#         sdf_vals_surface = np.zeros((self.num_sample_points//2, 1))
#         sdf_vals = np.vstack((sdf_vals_non_surface, sdf_vals_surface))


        
#         return shape_idx, positions, sdf_vals