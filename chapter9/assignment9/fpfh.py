#// /**
#//  * @file fpfh.py
#//  * @author brucesz
#//  * @date 2022/03/20 22:32:58
#//  * @version $Revision$
#//  * @brief
#//  *
#//  **/


import pandas as pd
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

import common


def compute_spfh(pcd, st, pt, radius, B):
    pts = np.asarray(pcd.points)
    norms = np.asarray(pcd.normals)

    k, idx_n, _ = st.search_radius_vector_3d(pt, radius)
    # pt's normal
    n1 = norms[idx_n[0]]

    idx_n = idx_n[1:]

    # (p2-p1)/norm(p2-p1)
    dis = pts[idx_n] - pt
    print("dtype is: ",dis.dtype)
    dis = dis/np.reshape(np.linalg.norm(dis, ord=2, axis=1), (k-1, 1))

    u = n1
    v = np.cross(u,dis)
    w = np.cross(u,v)

    n2 = norms[idx_n]
    
    # alpha: v*n2
    alpha = np.reshape((v*n2).sum(axis=1),(k-1,1))

    # phi: u * (p2-p1)/norm(p2-p1)
    phi = np.reshape((u*dis).sum(axis=1),(k-1,1))


    # theta: arctan(w*n2, u*n2)
    theta = np.reshape(np.arctan2((w*n2).sum(axis=1), (u * n2).sum(axis=1)),(k-1, 1))

    h_alpha = np.reshape(np.histogram(alpha, B, range=[-1.0,1.0])[0],(1,B))
    h_phi = np.reshape(np.histogram(alpha, B, range=[-1.0,1.0])[0],(1,B))
    h_theta = np.reshape(np.histogram(theta, B, range=[-1.0,1.0])[0],(1,B))

    spfh = np.hstack((h_alpha, h_phi, h_theta))
    return spfh


def compute_fpfh(pcd, st, kp1, radius, B):
    pts = np.asarray(pcd.points)
    k, idx_n, _ = st.search_radius_vector_3d(kp1, radius)
    print("k is: ",k)
    idx_n = idx_n[1:]
    print("shape of idxn: ",np.asarray(idx_n).shape)
    w = 1/np.linalg.norm(kp1 - pts[idx_n], ord=2, axis=1)
    print("shape of w: ",w.shape)
    neighbor_spfh = []
    for i in idx_n:
        spfh = compute_spfh(pcd, st, pts[i], radius, B)
        neighbor_spfh.append(spfh)
    n_spfh = np.array(neighbor_spfh)
    n_spfh = np.reshape(n_spfh, (k-1, 3*B))
    self_spfh  = compute_spfh(pcd, st, kp1, radius, B)

    fpfh = self_spfh + 1.0/(k-1) * np.dot(w, n_spfh)
    #n_spfh = np.reshape()
    return fpfh


def main():
    pcd_f = "/home/zhangs20/git/3d_process/dataset/modelnet40_normal_resampled/chair/chair_0001.txt"
    B=10
    radius=0.05
    pcd = common.read_pointcloud(pcd_f)

    st = o3d.geometry.KDTreeFlann(pcd)

    kp1 = np.array([0.4333,-0.7807, -0.4372])
    kp2 = np.array([-0.4240,-0.7850,-0.4392])
    kp3 = np.array([0.02323,0.004715,-0.2731])

    fpfh1 = compute_fpfh(pcd, st, kp1, radius, B)
    fpfh2 = compute_fpfh(pcd, st, kp2, radius, B)
    fpfh3 = compute_fpfh(pcd, st, kp3, radius, B)

    print("fpfh1 shape: ", fpfh1.shape)

    plt.plot(range(3*B), fpfh1.T, ls="-.", color='r', marker=",", lw=2, label="kp1")
    plt.plot(range(3*B), fpfh2.T, ls="-.", color='b', marker=",", lw=2, label="kp2")
    plt.plot(range(3*B), fpfh3.T, ls="-.", color='g', marker=",", lw=2, label="kp3")
    plt.show()


if __name__ == "__main__":
    main()













