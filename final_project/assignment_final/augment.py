#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2022 xxx.com, Inc. All Rights Reserved
#
########################################################################


import glob
import os
from os.path import split
import matplotlib.pyplot as plt
import shutil
import progressbar
import pandas as pd
import numpy as np
import open3d as o3d
import scipy

'''
@description:对点云进行随机旋转 
@param {*} points
@return {*}
'''
def random_rot(points):
        N = points.shape[0]

        points_ = points[:,0:3]
        normals_ = points[:,3:6]

        weights = scipy.spatial.distance.squareform(
            scipy.spatial.distance.pdist(points_, 'euclidean')
        ).mean(axis = 0)



        weights /= weights.sum()
        
        idx = np.random.choice(
            np.arange(N), 
            size = (64, ), replace=True if 64 > N else False,
            p = weights 
        )

        points_processed, normals_processed = points_[idx], normals_[idx]
        points = np.concatenate((points_processed,normals_processed),axis=1)
        theta = np.random.uniform(0, np.pi * 2)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        rot_points = points
        rot_points[:,0:2] = rot_points[:,0:2].dot(rotation_matrix)
        rot_points[:,3:5] = rot_points[:,3:5].dot(rotation_matrix)
        return rot_points

'''
@description: 读取点云
@param {*} bin_path
@return {*}
'''
def read_point_cloud(path):

        points = pd.read_csv(path)
        points = np.asarray(points)

        return points


def data_aug(path_class,N_aug,output_path):
        print(N_aug)
        progress = progressbar.ProgressBar()
        cls = path_class[0].split("/")[7]
        ##print("path class is: ",path_class[0])
        #print("path class is: ",path_class[0].split("/")[-1])
        #print("n_aug is: ", N_aug)
        #print("path class: ", len(path_class))
        index = 0
        for path in progress(path_class):
                # 读取点云
                points_source = read_point_cloud(path)
                if points_source.shape[0] < 4:
                        continue
                for _ in range(N_aug):
                        points_random = random_rot(points_source)
                        pd_points = pd.DataFrame(points_random)
                        o_path = os.path.join(output_path,f'{index:06d}.txt')
                        #raise RuntimeError("xx")
                        #print("output to: ",o_path)
                        pd_points.to_csv(o_path,index=False, header=None)
                        index = index + 1
        

def plot_dis(dataset_dir):
    path_pedestrian = glob.glob(os.path.join(dataset_dir,"pedestrian","*.txt"))
    path_cyclist = glob.glob(os.path.join(dataset_dir,"cyclist","*.txt"))
    path_vehicle = glob.glob(os.path.join(dataset_dir,"vehicle","*.txt"))
    path_misc = glob.glob(os.path.join(dataset_dir,"misc","*.txt"))
    N_pedestrian = len(path_pedestrian)
    N_cyclist = len(path_cyclist)
    N_vehicle = len(path_vehicle)
    N_misc = len(path_misc)

    plt.pie([N_pedestrian,N_cyclist,N_vehicle,N_misc],
            explode=None,
            labels=["pedestrian","cyclist","vehicle","misc"],
            colors=None,
            autopct='%1.1f%%')
    plt.show()


def do_aug(output_dir, dataset_dir):
    path_pedestrian = glob.glob(os.path.join(dataset_dir,"pedestrian","*.txt"))
    path_cyclist = glob.glob(os.path.join(dataset_dir,"cyclist","*.txt"))
    path_vehicle = glob.glob(os.path.join(dataset_dir,"vehicle","*.txt"))
    path_misc = glob.glob(os.path.join(dataset_dir,"misc","*.txt"))
    N_pedestrian = len(path_pedestrian)
    N_cyclist = len(path_cyclist)
    N_vehicle = len(path_vehicle)
    N_misc = len(path_misc)
    # 建立输出文件夹
    if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    # 对类别少的分类进行数据增强
    N_aug_pedestrian = int(N_vehicle/N_pedestrian)
    N_aug_cyclist = int(N_vehicle/N_cyclist)
    N_aug_misc = int(N_vehicle/N_misc)

    os.mkdir(os.path.join(output_dir,"pedestrian"))
    data_aug(path_pedestrian,N_aug_pedestrian,output_path=os.path.join(output_dir,"pedestrian"))
    os.mkdir(os.path.join(output_dir,"cyclist"))
    data_aug(path_cyclist,N_aug_cyclist,output_path=os.path.join(output_dir,"cyclist"))
    os.mkdir(os.path.join(output_dir,"misc"))
    data_aug(path_misc,N_aug_misc,output_path=os.path.join(output_dir,"misc"))
    os.mkdir(os.path.join(output_dir,"vehicle"))      
    data_aug(path_vehicle,1,output_path=os.path.join(output_dir,"vehicle"))
    # output_dir = "/home/teamo/point_process_piplines/KITTI/object/object_training_datasets"



if __name__ == "__main__":
        output_dir = "/mnt/e/download/kitti/class_train_aug"
        dataset_dir = "/mnt/e/download/kitti/class_train"
        
        plot_dis(output_dir)
        
        #do_aug(output_dir, dataset_dir)
        
       