import os
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

