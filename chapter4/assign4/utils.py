#!/usr/bin/env python

import os
import numpy as np


#deprecated
def get_pcd_per_cls_deprecate(p_dir):
        classes = os.listdir(p_dir)
        for cls_path in classes:
            cls_path = os.path.join(p_dir, cls_path)
            pcd_per_cls = os.listdir(cls_path)[0]
            pcd_path = os.path.join(cls_path, pcd_per_class)
            print(pcd_path)
            
            
def get_key(p):
    return p.split("/")[0]


def get_pcd_cls(p_dir):
  
    m_ = {}
    listf = os.path.join(p_dir, "filelist.txt")
    with open(listf,"r") as f:
        l = f.readlines()
        l = list(map(lambda x:x.strip(), l))
        for i in l :
            key = get_key(i)
            if not key in m_:
                m_[key] = os.path.join(p_dir, i)
    return m_


def pca(pts, number_pc= 1):
    #print("raw shape",pts.shape)
    assert pts.shape[1] == 3
    X = pts.transpose()
    x_bar = np.sum(X, axis=1)/len(pts)
    #print("x_bar",x_bar.transpose())
    #print("X shape: ", X.shape)
    center_X = (X.transpose() - x_bar).transpose()
    center_X_t = center_X.transpose()
    #print("x_bar_t",center_X_t.shape)
    H = np.matmul( center_X, center_X_t)
    #print(H.shape)
    U,s, V = np.linalg.svd(H)
    # default is descending
    target_eigen_vs = U[:,:number_pc]
    #print(" u shape: ", U.shape)
    #print("eigen values: ",s)
    #print("target eigen vec shape: ", target_eigen_vs.shape)
    # encode phase
    proj_res = np.matmul(target_eigen_vs.transpose(), X)
    #print("low dimension shape",proj_res.shape)
    reconv = np.matmul(target_eigen_vs, proj_res)
    #print("reconv shape: ",reconv.shape)
    return reconv.transpose(), U, s

def plot_pcd_from_path(path):
    pts = load_pcd(path)
    import open3d as o3d
    d_pcd = o3d.geometry.PointCloud()
    d_pcd.points = o3d.utility.Vector3dVector(pts[:,0:3])
    #d_pcd.colors = o3d.utility.Vector3dVector(pts[:,3:6])
    o3d.visualization.draw_geometries([d_pcd])

def plot_pcd_from_pts(pts):
    import open3d as o3d
    d_pcd = o3d.geometry.PointCloud()
    d_pcd.points = o3d.utility.Vector3dVector(pts[:,0:3])
    #d_pcd.colors = o3d.utility.Vector3dVector(pts[:,3:6])
    o3d.visualization.draw_geometries([d_pcd])
        
def load_pcd(path):
        import numpy as np
        pts = np.loadtxt(open(path,'r'), delimiter=',' , dtype = np.float32)
        pts = pts.reshape(-1, 6)
        return pts
        #d_pcd = o3d.geometry.PointCloud()
        #d_pcd.points = o3d.utility.Vector3dVector(pts[:,0:3])
        #d_pcd.colors = o3d.utility.Vector3dVector(pts[:,3:6])
        #o3d.visualization.draw_geometries([d_pcd])
        
        
def load_datas(dataset):
    res = {}
    for cls in dataset:
        path = dataset[cls]
        pts = load_pcd(path)
        res[cls] = pts
    return res


def plot_pcd_with_normal(pts):
    print("shape of pts: ", pts.shape)
    
    import open3d as o3d
    d_pcd = o3d.geometry.PointCloud()
    d_pcd.points = o3d.utility.Vector3dVector(pts[:,0:3])
    d_pcd.normals = o3d.utility.Vector3dVector(pts[:,3:6])
    #d_pcd.colors = o3d.utility.Vector3dVector(pts[:,3:6])
    o3d.visualization.draw_geometries([d_pcd],point_show_normal=True)

    
def print_class():
    pass


if __name__ == "__main__":
    p_dir = "./dataset/modelnet40_normal_resampled"
    classes = os.listdir(p_dir)
    print(classes)
    m_ = get_pcd_cls(p_dir)
    for k in m_:
        print("cls: ", k, " : ", m_[k])
    print(len(m_))