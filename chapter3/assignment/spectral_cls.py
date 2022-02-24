# 文件功能： 实现 K-Means 算法

import numpy as np
import sys
from pandas import CategoricalDtype

from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class Spectral_Clustering(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, k = 10 , dim = 2):
        self.n_cluster_ = n_clusters
        self.neighbor_ = k
        self.h_dim_ = dim
       

    def compute_dist(self, p1, p2):
       res = np.sum((p1-p2)**2)
       return res

    def compute_sim_matrix(self, X: np.ndarray):
        S = np.zeros((X.shape[0],X.shape[0]))
        for i in range(len(X)):
            for j in range(i+1,len(X)):
                S[i][j] = 1.0 *  self.compute_dist(X[i],X[j])
                S[j][i] = S[i][j]

        return S

    def compute_knn_adj_matrix(self, S: np.ndarray, k, sigma=1.0):
        N = len(S)
        A = np.zeros((N,N))

        for i in range(N):
            dist_index = zip(S[i],range(N))
            dist_index_s = sorted(dist_index, key = lambda x:x[0])
            #print(dist_index_s)
            n_index = [dist_index_s[m][1] for m in range(k+1)]
            for ix in n_index:
                A[i][ix] = np.exp(-S[i][ix]/2/sigma/sigma)
                A[ix][i] = A[i][ix]
        return A


    def compute_fc_adj_matrix(self, S: np.ndarray, sigma=1.0):
        N = len(S)
        A = np.zeros((N,N))

        for i in range(N):
            for j in range(i+1,N):
                A[i][j] = np.exp(-S[i][j]/2/sigma/sigma)
                A[j][i] = A[i][j]
        return A

    def compute_lapcian(self, adj_m: np.ndarray):
        d_m = np.sum(adj_m,axis=1)
        l_m = np.diag(d_m) - adj_m
        d_factor_m = np.diag(1.0/(d_m)**(0.5))
        return np.dot(np.dot(d_factor_m,l_m),d_factor_m)

    
    def fit(self, data):
        print("fit spectral clustering")
        print("type of data is: {}".format(data.shape))
        #print("type of data is: ", type(data), " shape of data is: ", data.shape)
        # 作业 spectral clustering
        # 屏蔽开始
        s_m = self.compute_sim_matrix(data)
        print("shape of s m: {}".format(s_m.shape))
        adj_m = self.compute_knn_adj_matrix(s_m, self.neighbor_)
        #adj_m = self.compute_fc_adj_matrix(s_m)
        l_m = self.compute_lapcian(adj_m)
        x,V = np.linalg.eig(l_m)
        print("shape of V is:",V.shape)
        #x_sort = np.argsort(x)
        x = zip(x,range(len(x)))
        x_s = sorted(x,key=lambda x:x[0])
        print(V[:,0].shape)
        #for (v,i) in (x_s):
        #    print(V[:,i].shape)
        self.H = np.vstack([V[:,i] for (v,i) in (x_s[:self.h_dim_])]).T
        print("shape of H is: {}".format(self.H.shape))

        self.km_ = KMeans(self.n_cluster_).fit(self.H)
        self.labels_ = self.km_.labels_

        print("fit done.")
        # 屏蔽结束

    def predict(self, p_datas):
        result = []
        # 作业2
        # 屏蔽开始
        # result = self.km_.predict(p_datas)
        raise RuntimeError("There is no predict func for spectral clustering")

        # 屏蔽结束
        return result
def data_gen():
    n_samples = 1500
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    no_structure = np.random.rand(n_samples, 2), None

    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = datasets.make_blobs(n_samples=n_samples,
                                cluster_std=[1.0, 2.5, 0.5],
                                random_state=random_state)

    da = [
    (noisy_circles, {'damping': .77, 'preference': -240,
                     'quantile': .2, 'n_clusters': 2,
                     'min_samples': 20, 'xi': 0.25}),
    (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
    (varied, {'eps': .18, 'n_neighbors': 2,
              'min_samples': 5, 'xi': 0.035, 'min_cluster_size': .2}),
    (aniso, {'eps': .15, 'n_neighbors': 2,
             'min_samples': 20, 'xi': 0.1, 'min_cluster_size': .2}),
    (blobs, {}),
    (no_structure, {})]
    return da


def example_test():
    
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    sc = Spectral_Clustering(n_clusters=2)
    col_set = ['r','g']
    sc.fit(x)
    
    x1 = []
    x2 = []
    for i,j in x:
        x1.append(i)
        x2.append(j)
   
    #cat = k_means.predict(x)
    cat = sc.labels_
    print(cat)

   
    colors = []
    for c in cat:
        colors.append(col_set[c])
    plt.scatter(x1,x2,c=colors)
    plt.show()


def circle_test():
    n_samples = 1500
    da = data_gen()
    dataset, algo_params = da[0]
    X, label = dataset
    #X = StandardScaler().fit_transform(X)
    print("shape of circles: {}".format(X.shape))
    col_set = ['r','g']
    x1 = X[:,0]
    x2 = X[:,1]
    #print(np.unique(np.array(y)))
    colors = []
    #X = StandardScaler().fit_transform(X)

    sc = Spectral_Clustering(n_clusters=2,k=10)
    sc.fit(X)
    #pred = sc.predict(X)
    pred = sc.labels_
    for i in pred:
        colors.append(col_set[i])
    #    print(i)

    
    plt.scatter(x1, x2, c= colors)
    plt.show()

if __name__ == '__main__':
    #example_test()
    circle_test()

