import numpy as np
import trimesh
import open3d as o3d
import os
import neuralImplicitTools.src.geometry as geometry

os.environ['PYOPENGL_PLATFORM'] = 'egl'

sample_surface = True
sample_cloud = True


# these should be paths to folders that contain a `model_normalized.obj`
PATHS = [
        "temp_plane_data/plane1/",
        "temp_plane_data/plane2/",
]

for prefix in PATHS:
    print("Loading trimesh")
    mesh = geometry.Mesh(meshPath= prefix + 'model_normalized.obj')

    if sample_surface:

        print("Sampling Surface")
        surface_sampler = geometry.PointSampler(mesh, ratio=0.0, std=0.0)
        surface_points = surface_sampler.sample(10_000)

        print("Saving surface numpy array")
        np.savez(prefix + "surface_points.npz", surface_points=surface_points)

        print("Saving surface point cloud")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(surface_points)
        o3d.io.write_point_cloud(prefix + "surface_points.ply", pcd)


    if sample_cloud:
        print("Making sdf")
        sdf = geometry.SDF(mesh)

        print("Sampling cloud")
        importanceSampler = geometry.ImportanceSampler(mesh, 10_000_000, 20)
        weighted_points = importanceSampler.sample(1_000_000)

        uniform_points = geometry.PointSampler(mesh, ratio=1.0, std=0.0).sample(100_000)

        cloud_points = np.concatenate( (weighted_points, uniform_points))


        print("Querying sdf")
        distances = sdf.query(cloud_points)

        print("Saving cloud numpy array")
        np.savez(prefix + "cloud_points.npz", cloud_points=cloud_points, distances=distances)

        print("Saving cloud point cloud")
        colors = np.stack((distances, distances, distances), axis=1)
        colors = colors.reshape(-1, 3)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud_points)
        o3d.io.write_point_cloud(prefix + "cloud_points.ply", pcd)

        inside = cloud_points[(distances < 0).squeeze()]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(inside)
        o3d.io.write_point_cloud(prefix + "inside_points.ply", pcd)

