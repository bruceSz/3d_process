#!/usr/bin/env python
import utils

import numpy as np
import open3d as o3d

def build_index(pts):
    """ build spatial index for this point cloud"""
    pass

def normal_estimate_entry():
    p_dir = "../dataset/modelnet40_normal_resampled"
    m_ = utils.get_pcd_cls(p_dir)
    da_map = utils.load_datas(m_)
    print(da_map.keys())
    target = "airplane"
    da = da_map[target]
    print(type(da))
    print(da.shape)
    da = da[:,0:3]

    pcd = o3d.geometry.PointCloud()

    pcd.points =o3d.utility.Vector3dVector(da)
    
    # for each pts in  da, compute normal for it .
    # TODO better knn
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    normals = []
    for idx in range(da.shape[0]):

        a = da[idx,:]
        [k, idx, _] = pcd_tree.search_knn_vector_3d(a,10)
        res = da[idx,:]
        #print(a)
        #print(da[1,:])
        #print(res)
        #print("k is: ",k)
        #print("idx is: ",idx)
        pca_res,U,s = utils.pca(res)
        normals.append(U[:,2])

    normal_arr = np.array(normals, dtype=np.float64)
    print(normal_arr.shape)
    new_pts = np.concatenate((da,normal_arr),axis=1)
    utils.plot_pcd_with_normal(new_pts)
    

    #for i in range(da.shape[0]):
    #    tmp = da[i,:]
    #    [] = pcd_tree.search_knn_vector_3d(tmp,10):




if __name__ == "__main__":
    normal_estimate_entry()

