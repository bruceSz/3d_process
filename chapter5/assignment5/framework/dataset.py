import torch
import os
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


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
    
    # TODO: rotation to feature

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
  train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
  cnt = 0
  print("DataLoader build done.")
  for pts, label in train_loader:
    print(pts.shape)
    print(label.shape)
    cnt += 1
    if cnt > 3:
      break    


if __name__ == "__main__":
  example_read_pcd()