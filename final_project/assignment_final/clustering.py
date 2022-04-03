# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类

import numpy as np
import os
import struct
from sklearn import cluster, datasets, mixture
from itertools import cycle, islice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import utils

from sklearn import cluster

# 功能：从kitti的.bin格式点云文件中读取点云
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




def lsq_compute_lane(data):
    A = np.hstack((data, np.ones((data.shape[0],1),dtype=np.float64)))
    AtA = np.matmul(A.T,A)
    print("AtA shape{}".format(AtA.shape))
    U, S, V = np.linalg.svd(AtA)
    print("shape of U {}".format(U.shape))
    # last U , if using A * A.T, chose v
    # normalize it before return.
    w = U[:,-1] / np.linalg.norm(U[:,-1])
    return w



def get_inner_idx(pts, plane, thr):
    '''
        refer: https://www.cnblogs.com/graphics/archive/2010/07/10/1774809.html
    '''
    N = pts.shape[0]
    da = np.hstack((pts, np.ones((N,1), dtype=np.float32)))
    print("shape of da: {}".format(da.shape))
    print("shape of plane {}".format(plane.shape))
    dis = abs(np.matmul(da,plane))/np.linalg.norm(plane[0:3])
    # for dis < thr, it it true.
    return dis < thr


def get_inner(pts, plane, thr):
    idxes = get_inner_idx(pts, plane, thr)
    return pts[idxes]


def compute_max_ransac_num(confi, inner_r, sample_n ):
    return int(np.ceil(np.log(1-confi)/np.log(1-inner_r**sample_n)))

def compute_plane_by_ransac(pts, thr, max_iter, max_inner_rate):
    # number of points
    N = pts.shape[0]

    # assume 0.6 points are plane point.
    max_inner_n = np.floor(N*max_inner_rate)
    inner_max_n = 0
    plane = np.zeros((4,1),dtype=float)

    for i in range(max_iter):
        # minimum number of points to fit a plane
        # no repeat choice
        rand_idx = np.random.choice(np.arange(N),size=3,replace=False)
        sample_pts = pts[rand_idx,:]
        p_ = lsq_compute_lane(sample_pts)
        inner_pts = get_inner(pts, p_, thr)
        inner_num = len(inner_pts)
        
        if inner_num > max_inner_n:
            #last round lsq fit, using all inner_pts as input
            # leave earlier.
            plane = lsq_compute_lane(inner_pts)
            break
        if inner_num > inner_max_n:
            plane = lsq_compute_lane(inner_pts)
            inner_max_num = inner_num

    inner_idx = get_inner_idx(pts, plane, thr)
    inner_pts = pts[inner_idx]

    plane = lsq_compute_lane(inner_pts)
    return plane, inner_idx, inner_pts

# 功能：从点云文件中滤除地面点
# 输入：
#     data: 一帧完整点云
# 输出：
#     segmengted_cloud: 删除地面点之后的点云
def ground_segmentation(data):
    # 作业1
    # 屏蔽开始
    confi = 0.99
    inner_r = .6
    max_iter = compute_max_ransac_num(confi, inner_r, 3)
    max_inner_ratio = inner_r * 0.7
    thr = 0.3
    #1. compute plane by ransac
    #2. compute point near plane, candidate_pts
    plane, inner_idx, inner_pts = compute_plane_by_ransac(data, thr, max_iter, max_inner_ratio)

    

    #3. remove candidate_pts from points, return.
    # data not in/near plane
    segmengted_cloud = data[~inner_idx]

    # 屏蔽结束
    print('origin data points num:', data.shape[0])
    print('segmented data points num:', segmengted_cloud.shape[0])
    return segmengted_cloud

# 功能：从点云中提取聚类
# 输入：
#     data: 点云（滤除地面后的点云）
# 输出：
#     clusters_index： 一维数组，存储的是点云中每个点所属的聚类编号（参考上一章内容容易理解）
def clustering(data):
    # 作业2
    # 屏蔽开始
    # estimate bandwidth for mean shift
    X = data
    print("clustering data with shape {}".format(X.shape))
    bandwidth = cluster.estimate_bandwidth(X, quantile=.3)
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_

    n_cls = np.unique(labels)

    #print(labels)

    # 屏蔽结束

    return labels

# 功能：显示聚类点云，每个聚类一种颜色
# 输入：
#      data：点云数据（滤除地面后的点云）
#      cluster_index：一维数组，存储的是点云中每个点所属的聚类编号（与上同）
def plot_clusters(data, cluster_index):
    ax = plt.figure().add_subplot(111, projection = '3d')
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(cluster_index) + 1))))
    colors = np.append(colors, ["#000000"])
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=2, color=colors[cluster_index])
    plt.show()


def main(seg_flag=True):
    root_dir = 'data/' # 数据集路径
    cat = os.listdir(root_dir)
    cat = cat[1:]
    iteration_num = len(cat)

    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        print('clustering pointcloud file:', filename)

        origin_points = read_velodyne_bin(filename)
        if seg_flag:
            segmented_points = ground_segmentation(data=origin_points)
        else:
            segmented_points = origin_points
        cluster_index = clustering(segmented_points)

        plot_clusters(segmented_points, cluster_index)


def plt_origin_pcd():
    test_p = 'data/000002.bin'
    arr = read_velodyne_bin(test_p)
    print(arr.shape)
    utils.plot_pcd_from_pts(arr)


def plot_pcd_with_different_color():
    test_p = 'data/000002.bin'
    arr = read_velodyne_bin(test_p)
    print(arr.shape)
    plane, inner_idx, inner_pts = compute_plane_by_ransac(arr)
    #utils.plot_pcd_from_pts(arr)
    
    plane_pcd = o3d.geometry.PointCloud()
    plane_pcd.points = o3d.utility.Vector3dVector(arr[inner_idx,0:3])
    plane_pcd.paint_uniform_color([0,1,0])
    #d_pcd.colors = o3d.utility.Vector3dVector(pts[:,3:6])
    
    o_pcd = o3d.geometry.PointCloud()
    o_pcd.points = o3d.utility.Vector3dVector(arr[~inner_idx,0:3])
    o_pcd.paint_uniform_color([1,0,0])

    o3d.visualization.draw_geometries([o_pcd, plane_pcd])



def tmp():
    test_p = 'data/000002.bin'
    arr = read_velodyne_bin(test_p)
    #segmented_points = ground_segmentation(data=arr)
    segmented_pts = arr[:100,:]
    print("pcd shape {}".format(segmented_pts.shape))
    cluster_index = clustering(segmented_pts)
    print(cluster_index)


if __name__ == '__main__':
    main(False)
