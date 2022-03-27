#// /**
#//  * @file icp.py
#//  * @author brucesz
#//  * @date 2022/03/27 22:30:00
#//  * @version $Revision$
#//  * @brief
#//  *
#//  **/

import os
import pandas as pd

import common
import iss
import fpfh

def get_pairs():
    data_root= "../../dataset/registration_dataset"
    reg_res_p = os.path.join(data_root,"reg_result.txt")
    assert(os.path.exists(reg_res_p))
    print(reg_res_p)
    df = pd.read_csv(reg_res_p)

    return df

def visualize_pair(pc1, pc2, trans):


def get_pc_path(name):
    data_root= "../../dataset/registration_dataset"
    pc_path = os.path.join(data_root, "point_clouds",name+".bin")
    return pc_path

def main():
    # get pair file
    p_df = get_pairs()
    radisu = 0.6
    B=10
    for idx, row in p_df.iterrows():
        s_pc_idx = row["idx1"]
        t_pc_idx = row["idx2"]
        s_pc = common.read_pointcloud(get_pc_path(s_pc_idx))
        t_pc = common.read_pointcloud(get_pc_path(t_pc_idx))

        #remove outlier
        s_pc, _ = s_pc.remove_radius_outlier(nb_points=4, radius=radius)
        t_pc,_  = t_pc.remove_radius_outlier(nb_points=4, radius=radius)

        # flann kd tree
        s_pc_kd = o3d.geometry.KDTreeFlann(s_pc)
        t_pc_kd = o3d.geometry.KDTreeFlann(t_pc)

        # detect feature points with iss
        _, s_kps = iss.det_kps(s_pc)
        _, t_kps = iss.det_kps(t_pc)

        s_kps_pc = s_pc.select_by_index(s_kps["idx"].values)
        t_kps_pc = t_pc.select_by_index(t_kps["idx"].values)

        # generate descriptors
        s_ds = []
        t_ds = []
        for p in s_kps_pc:
            fpfh1 = compute_fpfh(s_pc, s_pc_kd, p, radius, B)
            s_ds.append(fpfh1)

        for p in t_kps_pc:
            fpfh1 = compute_fpfh(t_pc, t_pc_kd, p, radius, B)
            t_ds.append(fpfh1)

        # init with ransac
        init_trans = ransac_init(s_kps_pc, s_ds, t_kps_pc, t_ds)

        # solve with icp main loop
        final_trans = icp_loop(s_pc, t_pc, init_trans)
        





    
   

if __name__ == "__main__":
    main()