import pandas as pd
import open3d as o3d
# 读取点云文件
def read_pointcloud(file_name):
    df = pd.read_csv(file_name,header = None)
    df.columns = ["x","y","z",
                    "nx","ny","nz"]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(df[["x","y","z"]].values)
    pcd.normals = o3d.utility.Vector3dVector(df[["nx","ny","nz"]].values)

    return pcd