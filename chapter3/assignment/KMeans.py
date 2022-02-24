# 文件功能： 实现 K-Means 算法

import numpy as np
import sys


class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    def compute_dist(self, p, centers):
        #print("shape of p: ",p.shape)
        #print("shape of center: ", centers.shape)
        dists = [np.linalg.norm(p - centers[i]) for i in range(len(centers))]
        cls = dists.index(min(dists))
        return cls, min(dists)

    def fit(self, data):
        print("type of data is: ", type(data), " shape of data is: ", data.shape)
        # 作业1
        # 屏蔽开始
        #1. init centers first k element as center of cluster
        self.centers_ = []
        for i in range(self.k_):
            self.centers_.append(data[i])
        self.centers_ = np.array(self.centers_)
        #data = data[self.k_:]
        
        # iterate max_iter_, e an m step 
        for i in range(self.max_iter_):
            
            assigment = {}
            for i in range(self.k_):
                assigment[i] = []
            # e_step
            # for each p in data, compute distance against each center , find the ith center with minmum
            # `distance` and assign p to i
            for idx in range( len(data)):
                minv = sys.maxsize
                min_idx = -1
                #for tmpi in range(self.k_):
                    #dist =  self.compute_dist(data[i],self.centers_[tmpi])
                    #if dist < minv:
                    #    minv = dist
                    #    min_idx = tmpi
                cls, _  = self.compute_dist(data[idx], self.centers_)
                assigment[cls].append(data[idx])

            for c in assigment:
                assigment[c] = np.array(assigment[c])
                print("shape of assignment is: ", assigment[c].shape)
                

            # m_step
            # update center
            prev_centers = self.centers_
            for c in assigment:
                self.centers_[i] = np.average(assigment[c], axis=0)
            need_opt= False
            for c in range(len(self.centers_)):
                old_center = prev_centers[c]
                new_center = self.centers_[c]
                if (np.sum((new_center - old_center)/old_center) * 100) > self.tolerance_:
                    need_opt = True
            if not need_opt:
                break



        # 屏蔽结束

    def predict(self, p_datas):
        result = []
        # 作业2
        # 屏蔽开始
        for p in p_datas:
            cls, dist = self.compute_dist(p, self.centers_)
            result.append(cls)


        # 屏蔽结束
        return result

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k_means = K_Means(n_clusters=2)
    col_set = ['r','g']
    k_means.fit(x)
    x1 = []
    x2 = []
    for i,j in x:
        x1.append(i)
        x2.append(j)
   
    cat = k_means.predict(x)

   
    colors = []
    for c in cat:
        colors.append(col_set[c])
    plt.scatter(x1,x2,c=colors)
    plt.show()
    #print(cat)

