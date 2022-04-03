#!/usr/bin/env python

import os
import glob
import pandas as pd
import random
import numpy as np


def get_train_val_test():
    train_p = "train_val_set/ImageSets/train.txt"
    val_p = "train_val_set/ImageSets/val.txt"
    test_p = "train_val_set/ImageSets/test.txt"
    train = []
    val = []
    test = []
    with open(train_p,"r") as f:
        for l in f:
            train.append(l.strip())
    
    with open(val_p,"r") as f:
        for l in f:
            val.append(l.strip())

    with open(test_p,"r") as f:
        for l in f:
            test.append(l.strip())
   # print(len(train))
    #print(train[:10])
    return train, val, test


def gen_sim_val_det_result():
    gt_path  = glob.glob("data/object/label_2/*.txt")
    #

    _, val, _ = get_train_val_test()
    val_set = set(val)
    out_path = "./data/det"
    da_list = []
    print("val set size: ", len(val_set))
    for p in gt_path:
        name = p.split("/")[-1].split(".")[0]
        if name in val_set:
            #print(p)
            da = pd.read_csv(p, sep=" ", header=None)
            da.columns = ["type",
                "truncated",
                "occluded",
                "alpha",
                "bbox_left","bbox_top","bbox_right","bottom",
                    "height","width","length",
                "x","y","z",
                "rotation_y"]
            # generate random score
            da["score"] = np.random.random(da.shape[0])
            #da.to_csv("results/xx.txt")
            # for other type detect as car 
            da.loc[(da["type"] != "Car") & (da["type"] != "Cyclist") & (da["type"] != "Pedestrian"), ["type"]] = "Car"
            print("to_csv: ","results/result_ha/data/"+ p.split("/")[3] )
            da.to_csv("./results/result_ha/data/"+ p.split("/")[3] ,sep = " ",header = False ,index = False)
            #da.to_csv("x")
            #da_list.append(da)
        #print(da)
    #print(len(da_list))


if __name__ == "__main__":
    #main()
    #get_train_val_test()
    gen_sim_val_det_result()