## Notes

### Overview

1. Get shapenet obj meshes
2. Make shapenet meshes manifold using [ManifoldPlus](https://github.com/hjwdzh/ManifoldPlus) (do this on your own).
3. Turn meshes into point clouds with signed distance values via fast winding number method from [Weight-Encoded Neural Implicits](https://github.com/u2ni/ICML2021) paper. (use `scan_points.py` script)
4. train model (use `train.py train`)
5. eval model (use `train.py eval`)


### Nitty Gritty

Creating pointclouds to train on
```
conda create --name scan_points python=3.6 trimesh open3d numpy  -c conda-forge -c pyg -c pytorch -c anaconda -c open3d-admin
conda activate scan_points

# now go into `scan_points.py` and add the folders that contain the `model_normalized.obj` that you want to turn into a point cloud.
# You can get these from running the ManifoldPlus tool on shapenet meshes.
# (You will need to modify a variable called `PATHS` in `scan_points.py`).

# now you can run the script
python scan_points
```


Setup for training
```
conda create --name shapeformer python=3.7 tensorboard trimesh scikit-learn open3d pytorch-lightning pytorch torchvision torchaudio cudatoolkit=10.2 einops numpy pytorch-scatter -c conda-forge -c pyg -c pytorch -c anaconda -c open3d-admin

conda activate shapeformer

~/miniconda3/envs/shapeformer/bin/pip install PyMCubes

# if the miniball directory is empty you may need to clone it from https://github.com/weddige/miniball
cd miniball 
python setup.py install
cd ..
```

Training
```
conda activate shapeformer

# now go into the `train.py` script and modify the `PATHS` variable so it has the folders that contain your point clouds from earlier.

python train.py train # train
python train.py eval --checkpoint lightning_logs/version_28/checkpoints/epoch=9999-step=39999.ckpt --descriptor data/airplanes/133937bd45f953748be6919d4632fec1/surface_points.npz # evaluate (find output at `out/mesh_0.obj`)
```

## Original
This is a simplified code of the ShapeFormer project, which contains model code and hyperparameters for the core models.

The main file for ShapeFormer and VQDIF models are in 'shapeformer/shapeformer.py' and 'vqdif/vqdif.py' respectively.
Their configuration yaml files are in this directoy.

We will release the complete code and pretrained model according to final decision.
