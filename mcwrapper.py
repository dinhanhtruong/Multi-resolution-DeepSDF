import numpy as np
import trimesh
import mcubes

USE_MARCHING_CUBES = True

def extract_mesh_mcubes(hashtable, model, save_path, occupancy=False,):
    # # load the descriptor to sample from disk
    # surface_points = np.load(DESCRIPTOR)["surface_points"]
    # descriptor = surface_points.reshape((1, -1, 3))
    # descriptor = torch.from_numpy(descriptor).float().to(device)

    # Pick a uniform grid to sample from
    resolution = 150
    xs, ys, zs = np.meshgrid(
        np.linspace(0, 1, resolution),
        np.linspace(0, 1, resolution),
        np.linspace(0, 1, resolution)
    )
    if USE_MARCHING_CUBES:
        points_to_sample = np.stack([xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)], axis=1)
        points_to_sample = points_to_sample.reshape(-1,3)
    # else:
    #     i = 0
    #     points_to_sample = np.zeros((resolution**3, 3))
    #     for x in range(resolution):
    #         for y in range(resolution):
    #             for z in range(resolution):
    #                 points_to_sample[i] = np.array([xs[x, y, z], ys[x, y, z], zs[x, y, z]])
    #                 i += 1

    #     points_to_sample = points_to_sample.reshape(1,-1,3)

    encoded_positions = hashtable(points_to_sample)
    preds = -model(encoded_positions).numpy()
    print("pos:", points_to_sample[:20])
    print("preds: ", preds[:20])

    points_to_sample = points_to_sample.reshape(-1,3)

    if USE_MARCHING_CUBES:
        distance_from_origin = np.sqrt(np.sum((points_to_sample - 0.5) ** 2, axis=1))
        # unit sphere
        preds[distance_from_origin > 0.5] = -1
        preds = preds.reshape(resolution, resolution, resolution)

    # choose isosurface
    THRESHOLD = 0

    if USE_MARCHING_CUBES:
        vertices, triangles = mcubes.marching_cubes(preds, THRESHOLD)
        mcubes.export_obj(vertices, triangles, save_path+'.obj' )
        # mcubes.export_obj(vertices, triangles, f'./out/mesh_{i}.obj')
    # else:
    #     reconstructed_point_cloud = points_to_sample[preds > THRESHOLD]

    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(reconstructed_point_cloud)
    #     o3d.io.write_point_cloud(f"./out/data_{i}.ply", pcd)