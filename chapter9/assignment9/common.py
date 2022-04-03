import pandas as pd
import open3d as o3d
import os
import numpy as np

# 读取点云文件
def read_pointcloud(file_name):
    assert(os.path.isfile(file_name))
    print(file_name)
    data = np.fromfile(file_name, dtype=np.float32)
    data = data.reshape((-1, 6))
    print(data.shape)
    #df.columns = ["x","y","z",
    #                "nx","ny","nz"]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:, 0:3])
    pcd.normals = o3d.utility.Vector3dVector(data[:, 3:6])

    return pcd