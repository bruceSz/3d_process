#!/usr/bin/env python


import open3d as o3d
import numpy as np
import heapq
import pandas as pd

# load pcd file in modelnet40
def load_pcd(path):
        import numpy as np
        pts = np.loadtxt(open(path,'r'), delimiter=',' , dtype = np.float32)
        pts = pts.reshape(-1, 6)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[:,:3])
        pcd.normals = o3d.utility.Vector3dVector(pts[:,3:6])

        return  pcd


def visualize_pcd_with_kp(pcd, keypoints):
    pcd.paint_uniform_color([0.95,0.95,0.95])
    print("shape of keypoints:", keypoints["idx"])
    np.asarray(pcd.colors)[keypoints["idx"].values, :] = [1.0, 0.0, 0.0]
    o3d.visualization.draw_geometries([pcd])




def det_kps(pcd):
    Radius = 0.1
    Gamma1 = 1.2
    Gamma2 = 1.2
    MinEigen =  -1.0
    Kmin = 1


    neighbor_num_map = {}
    point_neighbors = {}
    search_tree = o3d.geometry.KDTreeFlann(pcd)

    #p_neighbors = np.zeros(())
    pts = np.asarray(pcd.points)
    res = []
    pq = []
    for idx_p, p in enumerate(pts):
        #print("point :",p)
        #print("shape of point, ",p," type of p: ",type(p.points))
        [k, idx, _] = search_tree.search_radius_vector_3d(p, Radius)

        w = []
        point_neighbors[idx_p] = idx
        idx = np.asarray(idx)

        for i in idx:
            # for each neighbor.
            if not i in neighbor_num_map:
                [k_, _, _] = search_tree.search_radius_vector_3d(pts[i], Radius)
                neighbor_num_map[i] = k_

            w.append(neighbor_num_map[i])
        #print("shape of pcd points:", idx.shape)
        neighbors = pts[idx]
        distance = neighbors - p
        w = np.asarray(w)
        w = np.reshape(w,(-1,))

        cov = 1.0/(w.sum()) * np.dot(distance.T, np.dot(np.diag(w), distance))
        e_v, _ = np.linalg.eig(cov)

        e_v = e_v[np.argsort(e_v)[::-1]]
        lambda3 = (e_v[-1])

        res.append([idx_p, e_v[0], e_v[1], e_v[2], k])
        heapq.heappush(pq, (-e_v[2], idx_p))


    nms = set([])
    while(pq):
        _, idx_p = heapq.heappop(pq)
        if not idx_p in nms:
            # push all neighbor into the nms set.
            n_ = point_neighbors[idx_p]
            # exclude it-self
            n_ = n_[1:]
            for _i in n_:
                nms.add(_i)


    res_df = pd.DataFrame(res, columns=["idx","l1", "l2", "l3","n_neighbors"])

    print("shape of res_df: ",len(res_df))
    
    df  = res_df.loc[res_df["idx"].apply(lambda id: not id in nms), res_df.columns]
    
    # found some l1 is complex number.
    df["l1"] = df["l1"].astype(float)

    print("shape of df after nms: ",len(df))
    print("min  of df  l1: ", min(df["l1"]))
    print("max  of df  l1: ", max(df["l1"]))
    df = df.loc[(df["l1"]>df["l2"]*Gamma1) & (df["l2"]>df["l3"]*Gamma2) & (df["l3"]>MinEigen), df.columns]
    print("shape of final df: ",len(df))
    return pcd, df






def main():
    test_data = "../../dataset/modelnet40_normal_resampled/bottle/bottle_0001.txt"
    pcd = load_pcd(test_data)

    pcd, kps = det_kps(pcd)

    visualize_pcd_with_kp(pcd, kps)


    pass

if __name__ == "__main__":
    main()
