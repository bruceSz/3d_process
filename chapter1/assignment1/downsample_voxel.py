#!/usr/bin/env python
import utils
import math
import numpy as np
import random
import open3d as o3d


def centroid(arr):
    center = np.mean(arr,axis=0)
    return center

def random_s(arr):
    idx = random.randint(0,len(arr)-1)
    return arr[idx,:]

def downsample_voxel(algo=centroid):
    voxel_size = 0.05
    p_dir = "../dataset/modelnet40_normal_resampled"
    m_ = utils.get_pcd_cls(p_dir)
    
    data = utils.load_datas(m_)
    print(data.keys())
   # for target in da.keys():
   #  '', '', '', '', '', 
   #   '', '', '', '', '', '', '', '', '', 'xbox']
    for target in ["flower_pot"]:
        # compute x,y,z axis scale
        
        da = data[target]
        print("pre downsample shape: ", da.shape)
        x_arr = da[:,0]
        y_arr = da[:,1]
        z_arr = da[:,2]

        min_x = np.min(x_arr)
        max_x = np.max(x_arr)

        min_y = np.min(y_arr)
        max_y = np.max(y_arr)

        min_z = np.min(z_arr)
        max_z = np.max(z_arr)

        dims = (math.ceil((max_x-min_x)/voxel_size), math.ceil((max_y-min_y)/voxel_size), math.ceil((max_z-min_z)/voxel_size))

        index_list = []

        for da_idx in range(da.shape[0]):
            tmp_da = da[da_idx,:]
            
            h_x = np.floor((tmp_da[0] - min_x) / voxel_size)
            h_y = np.floor((tmp_da[1] - min_y) / voxel_size)
            h_z = np.floor((tmp_da[2] - min_z) / voxel_size)
            index_ = h_x + h_y * dims[0] + h_z * dims[0] * dims[1]
            index_list.append(index_)

        index_arr = np.array(index_list)
        sorted_index_index = index_arr.argsort()
        sorted_index_arr = index_arr[sorted_index_index]
        curr_voxel = []
        res_pcd = []
        # sort da accoring to it's voxel-grid index
        sorted_da = da[sorted_index_index]
        for idx in range(sorted_da.shape[0]):
            if idx == ( sorted_da.shape[0] -1 ) or sorted_index_arr[idx] != sorted_index_arr[idx+1]:
                curr_voxel.append(sorted_da[idx,:])
                tmp_arr = np.array(curr_voxel)
                res = algo(tmp_arr[:,0:3])
                res_pcd.append(res)
                curr_voxel = []
            else:
                curr_voxel.append(sorted_da[idx,:])

        res_pcd = np.array(res_pcd)
        print("downsample result size:", res_pcd.shape)
        #return res_pcd
        utils.plot_pcd_from_pts(res_pcd)


      


    #for i in range(da.shape[0]):
    #    tmp = da[i,:]
    #    [] = pcd_tree.search_knn_vector_3d(tmp,10):




if __name__ == "__main__":
    downsample_voxel()
    downsample_voxel(algo=random_s)

