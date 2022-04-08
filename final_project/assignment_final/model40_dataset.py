import torch
import os
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd
import glob

import pickle

def read_pcd_from_file(file):
    np_pts = np.zeros(0)
    with open(file, 'r') as f:
        pts = []
        for line in f:
            one_pt = list(map(float, line[:-1].split(',')))
            pts.append(one_pt[:3])
        np_pts = np.array(pts)
    return np_pts


def read_file_names_from_file(file):
  with open(file, 'r') as f:
    files = []
    for line in f:
      files.append(line.split('\n')[0])
  return files


class PointNetDataset(Dataset):
  def __init__(self, root_dir, train):
    super(PointNetDataset, self).__init__()

    self._train = train
    self._classes = []

    self._features = []
    self._labels = []

    self.load(root_dir)

  def classes(self):
    return self._classes

  def __len__(self):
    return len(self._features)
  
  def __getitem__(self, idx):
    feature, label = self._features[idx], self._labels[idx]
    
    # TODO: normalize feature
    #print("feature input shape: {}".format(feature.shape))
    feature = feature - np.expand_dims(np.mean(feature, axis=0),0)
    #print("feature minus mean shape: {}".format(feature.shape))
    r = np.sqrt(np.sum(feature **2, axis=1))
    dist = np.max(r,0)
    #print("sigma shape: {}".format(dist))
    feature = feature/ dist
    
    # TODO: rotation to feature
    theta = np.random.uniform(0, np.pi*2)
    rm = np.array([
      [np.cos(theta), -np.sin(theta)],
      [np.sin(theta), np.cos(theta)]
    ])
    feature[:,[0,2]] = feature[:,[0,2]].dot(rm)

    # jitter
    feature += np.random.normal(0, 0.02, size=feature.shape)
    feature = torch.Tensor(feature.T)

    l_lable = [0 for _ in range(len(self._classes))]
    l_lable[self._classes.index(label)] = 1
    label = torch.Tensor(l_lable)

    return feature, label
  
  def load(self, root_dir):
    things = os.listdir(root_dir)
    files = []
    for f in things:
      if self._train == 0:
        if f == 'modelnet40_train.txt':
          files = read_file_names_from_file(root_dir + '/' + f)
      elif self._train == 1:
        if f == 'modelnet40_test.txt':
          files = read_file_names_from_file(root_dir + '/' + f)
      if f == "modelnet40_shape_names.txt":
        self._classes = read_file_names_from_file(root_dir + '/' + f)
    tmp_classes = []

    # debug purpose
    #import random
    #files = random.choices(files, k=100) 
    with tqdm(total = len(files)) as qbar:
      for file in files:
        num = file.split("_")[-1]
        kind = file.split("_" + num)[0]
        if kind not in tmp_classes:
          tmp_classes.append(kind)
        pcd_file = root_dir + '/' + kind + '/' + file + '.txt'
        np_pts = read_pcd_from_file(pcd_file)
        # print(np_pts.shape) # (10000, 3)
        self._features.append(np_pts)
        self._labels.append(kind)
        qbar.update(1)

    if self._train == 0:
      print("There are " + str(len(self._labels)) + " trian files.")
    elif self._train == 1:
      print("There are " + str(len(self._labels)) + " test files.")
      


def example_read_pcd():
  p = "../../../dataset/modelnet40_normal_resampled/airplane/airplane_0100.txt"
  pts = read_pcd_from_file(p)
  print("pts shape {}".format(pts.shape))



def example_dataset():
  print("shape of curr: ", os.getcwd())
  train_data = PointNetDataset(os.path.join(os.getcwd(),"../../../","./dataset/modelnet40_normal_resampled"), train=0)
  print("train data shape.")
  train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
  cnt = 0
  print("DataLoader build done.")
  for pts, label in train_loader:
    #print(pts.shape)
    print(label)
    cnt += 1
    #if cnt > 3:
    #  break    


if __name__ == "__main__":
  #example_read_pcd()
  example_dataset()
  