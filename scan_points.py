import numpy as np
import trimesh
import open3d as o3d
import os
import neuralImplicitTools.src.geometry as geometry

os.environ['PYOPENGL_PLATFORM'] = 'egl'

sample_surface = True
sample_cloud = True
get_all_points = True


# these should be paths to folders that contain a `model_normalized.obj`
PATHS = [
        
        # "temp_plane_data/cube_60w/",
        # "temp_plane_data/sphere/",
        # "temp_plane_data/plane1/",
        "temp_plane_data/plane2/"
]

TOTAL_NUM_SAMPLE_POINTS = 2**20
EXPONENTIAL_SAMPLE_WEIGHT = 60 # 20  higher = more concentrated near surface, optimal ~= 60

for prefix in PATHS:
    print("Loading trimesh")
    mesh = geometry.Mesh(meshPath= prefix + 'model_normalized.obj')

    if sample_surface:

        print("Sampling Surface")
        surface_sampler = geometry.PointSampler(mesh, ratio=0.0, std=0.0)
        surface_points = surface_sampler.sample(TOTAL_NUM_SAMPLE_POINTS//2)
        surface_distances = np.zeros((TOTAL_NUM_SAMPLE_POINTS//2, 1))


        print("Saving surface numpy array")
        np.savez(prefix + "surface_points.npz", surface_points=surface_points, surface_distances=surface_distances)

        print("Saving surface point cloud")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(surface_points)
        o3d.io.write_point_cloud(prefix + "surface_points.ply", pcd)


    if sample_cloud:
        NUM_CLOUD_POINTS = TOTAL_NUM_SAMPLE_POINTS//2
        print("Making sdf")
        sdf = geometry.SDF(mesh)

        print("Sampling cloud")
        importanceSampler = geometry.ImportanceSampler(mesh, 10_000_000, EXPONENTIAL_SAMPLE_WEIGHT)
        weighted_points = importanceSampler.sample((NUM_CLOUD_POINTS*7) // 8)
        uniform_points = geometry.PointSampler(mesh, ratio=1.0, std=0.0).sample(NUM_CLOUD_POINTS//8)

        cloud_points = np.concatenate( (weighted_points, uniform_points))
        

        print("Querying sdf")
        cloud_distances = sdf.query(cloud_points)

        print("Saving cloud numpy array")
        np.savez(prefix + "cloud_points.npz", cloud_points=cloud_points, cloud_distances=cloud_distances)


        print("Saving cloud point cloud")
        colors = np.stack((cloud_distances, cloud_distances, cloud_distances), axis=1)
        colors = colors.reshape(-1, 3)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud_points)
        o3d.io.write_point_cloud(prefix + "cloud_points.ply", pcd)

        inside = cloud_points[(cloud_distances < 0).squeeze()]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(inside)
        o3d.io.write_point_cloud(prefix + "inside_points.ply", pcd)

    if get_all_points:
        all_points = np.concatenate( (weighted_points, uniform_points, surface_points))
        all_distances = np.concatenate( (cloud_distances, surface_distances))

        rand_perms = np.random.permutation(TOTAL_NUM_SAMPLE_POINTS)
        all_points = all_points[rand_perms]
        all_distances = all_distances[rand_perms]

        np.savez(prefix + "all_points.npz", points=all_points, distances=all_distances)