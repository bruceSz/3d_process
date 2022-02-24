# -*- coding: utf-8 -*-
import struct
import matplotlib
import numpy as np
import os
import utils

# 输入：
#     path: 文件路径
# 输出：
#     点云数组
def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)


def plot_velodyne_bin(arr):
    pass



def load_pcd(path):
    np.fromfile(path,dtype=np.int32)

def plot_pcd_from_path(path):
    pts = load_pcd(path)
    import open3d as o3d
    d_pcd = o3d.geometry.PointCloud()
    d_pcd.points = o3d.utility.Vector3dVector(pts[:,0:3])
    #d_pcd.colors = o3d.utility.Vector3dVector(pts[:,3:6])
    o3d.visualization.draw_geometries([d_pcd])

    
def main():
    print("xxx")
    p1 = "./data" + "/" + "000000.bin"
    assert os.path.exists(p1)
    arr = read_velodyne_bin(p1)
    print("shape of arr {}".format(arr.shape))
    utils.plot_pcd_from_pts(arr)
    
if __name__ == "__main__":
    main()