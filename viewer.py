import trimesh
import pyrender

stl = trimesh.load('out2.stl', 'stl')
mesh = pyrender.Mesh.from_trimesh(stl)
scene = pyrender.Scene()
scene.add(mesh)
pyrender.Viewer(scene, use_raymond_lighting=True)