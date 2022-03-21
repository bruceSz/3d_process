#!/usr/bin/env python

import pandas as pd
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import pyshot


# 读取点云文件
def read_pointcloud(file_name):
    df = pd.read_csv(file_name,header = None)
    df.columns = ["x","y","z",
                    "nx","ny","nz"]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(df[["x","y","z"]].values)
    pcd.normals = o3d.utility.Vector3dVector(df[["nx","ny","nz"]].values)

    return pcd


def compute_shot(pts, faces, st, pt):
    #radius = [0.1, 1., 2.]
    radius = 0.1

    local_rf_radius = radius
    #n_bins = [10,20,100]
    n_bins = 10
    #min_neighbors= [3, 5, 6]
    min_neighbors= 3
    double_volumes_sectors = [True, False]
    use_interpolation = [True, False]
    use_normalization = [True, False]

    k, idx_n, _ = st.search_radius_vector_3d(pt, radius)
    idx_n = idx_n[1:]

    print("pts shape: ", pts.shape)
    verts = pts[idx_n]
    norms = faces[idx_n]
    

    shot_features = pyshot.get_descriptors(
                verts,
                norms,
                radius=radius,
                local_rf_radius=radius,
                min_neighbors=min_neighbors,
                n_bins=n_bins,
                double_volumes_sectors=True,
                use_interpolation=True,
                use_normalization=True,
    )

    

    assert np.isfinite(shot_features).all()

    n_features = 16 * (n_bins + 1) * (double_volumes_sectors + 1)
    assert shot_features.shape[1] == n_features
    return n_features

def main():
    #parser = argparse.ArgumentParser("shot", description=__doc__)
    
    pcd_f = "/home/zhangs20/git/3d_process/dataset/modelnet40_normal_resampled/chair/chair_0001.txt"
    
    pcd = read_pointcloud(pcd_f)
    pts =  np.asarray(pcd.points)
    norms = np.asarray(pcd.normals)

    st = o3d.geometry.KDTreeFlann(pcd)

    kp1 = np.array([0.4333,-0.7807, -0.4372])
    kp2 = np.array([-0.4240,-0.7850,-0.4392])
    kp3 = np.array([0.02323,0.004715,-0.2731])

    fpfh1 = compute_shot(pts, norms, st,  kp1)
    fpfh2 = compute_shot(pts, norms, st,  kp2)
    fpfh3 = compute_shot(pts, norms, st,  kp3)

    print("fpfh1 shape: ", fpfh1.shape)

    plt.plot(range(3*B), fpfh1.T, ls="-.", color='r', marker=",", lw=2, label="kp1")
    plt.plot(range(3*B), fpfh2.T, ls="-.", color='b', marker=",", lw=2, label="kp2")
    plt.plot(range(3*B), fpfh3.T, ls="-.", color='g', marker=",", lw=2, label="kp3")
    plt.show()



if __name__ == "__main__":
    main()