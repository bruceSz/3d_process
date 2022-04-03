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
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

import common
import iss
import fpfh
from fpfh import compute_fpfh
from main_logic import ransac_match
from main_logic import exact_match
from main_logic import get_ransac_guess

def get_pairs():
    data_root= "../../dataset/registration_dataset"
    reg_res_p = os.path.join(data_root,"reg_result.txt")
    assert(os.path.exists(reg_res_p))
    print(reg_res_p)
    df = pd.read_csv(reg_res_p)

    return df


def get_output():
    pass

def visualize_pair(pc1, pc2, trans):
    pass

def get_pc_path(name):
    data_root= "../../dataset/registration_dataset"
    assert(os.path.isdir(data_root))
    print("name: ", name)
    pc_path = os.path.join(data_root, "point_clouds",name+".bin")
    return pc_path


def format_output( idx1, idx2, T):
    """
    Add record to output
    """
    def format_transform_matrix(T):
        r = R.from_matrix(T[:3, :3])
        q = r.as_quat()
        t = T[:3, 3]

        return (t, q)

    res = []
    res.append(idx1)
    res.append(idx2)
    
    (t, q) = format_transform_matrix(T)

    # translation:
    res.append(t[0])
    res.append(t[1])
    res.append(t[2])
    # rotation:
    res.append(q[3])
    res.append(q[0])
    res.append(q[1])
    res.append(q[2])
    return res

def main():
    # get pair file
    p_df = get_pairs()
    radius = 0.6
    B=10

    df = []
    columns = ["idx1","idx2","t_x", "t_y","t_z","q_w","q_x","q_y","q_z"]

    for idx, row in p_df.iterrows():
        s_pc_idx = row["idx1"]
        t_pc_idx = row["idx2"]
        s_pc = common.read_pointcloud(get_pc_path(str(int(s_pc_idx))))
        t_pc = common.read_pointcloud(get_pc_path(str(int(t_pc_idx))))

        #remove outlier
        s_pc, _ = s_pc.remove_radius_outlier(nb_points=4, radius=radius)
        t_pc,_  = t_pc.remove_radius_outlier(nb_points=4, radius=radius)

        # flann kd tree
        s_pc_kd = o3d.geometry.KDTreeFlann(s_pc)
        t_pc_kd = o3d.geometry.KDTreeFlann(t_pc)

        # detect feature points with iss
        _, s_kps = iss.det_kps(s_pc)
        _, t_kps = iss.det_kps(t_pc)

        print("source kps shape: ", s_kps.shape)
        print("targer kps shape: ", t_kps.shape)
        

        s_kps_pc = s_pc.select_by_index(s_kps["idx"].values)
        t_kps_pc = t_pc.select_by_index(t_kps["idx"].values)

        # generate descriptors
        s_ds = []
        t_ds = []
        for p in s_kps_pc.points:
            fpfh1 = compute_fpfh(s_pc, s_pc_kd, p, radius, B)
            s_ds.append(fpfh1)

        for p in t_kps_pc.points:
            fpfh1 = compute_fpfh(t_pc, t_pc_kd, p, radius, B)
            t_ds.append(fpfh1)

        s_ds = np.asarray(s_ds)
        t_ds = np.asarray(t_ds)
        # init with ransac
        T = get_ransac_guess(s_kps_pc, t_kps_pc, s_ds, t_ds)
        fit_result, new_T = exact_match(s_pc, t_pc, t_pc_kd, T)

        df.append(format_output(s_pc_idx, t_pc_idx, new_T))

        # solve with icp main loop
        #final_trans = icp_loop(s_pc, t_pc, init_trans)


        # add to output as reg_result.txt
        #main_logic.dump_res(final_trans)
        break
    df = pd.DataFrame(df, columns=columns)
    df["idx1"] = df["idx1"].astype(int)
    df["idx2"] = df["idx2"].astype(int)
    df.to_csv("./reg_result_0403.txt", index=False)

    
   

if __name__ == "__main__":
    main()