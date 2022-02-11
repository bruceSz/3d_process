# 文件功能： 实现 K-Means 算法

import numpy as np
import sys

from sklearn.cluster import KMeans


class Spectral_Clustering(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, k=2, dim = 2):
        self.n_cluster_ = n_clusters
        self.neighbor_ = k
        self.h_dim_ = dim
       

    def compute_dist(self, p1, p2):
       res = np.sum((p1-p2)**2)
       #res = np.sqrt(res)
       return res

    def compute_sim_matrix(self, X: np.ndarray):
        S = np.zeros((X.shape[0],X.shape[0]))
        for i in range(len(X)):
            for j in range(i+1,len(X)):
                S[i][j] = 1.0 * self.compute_dist(X[i],X[j])
                S[j][i] = S[i][j]

        return S

    def compute_knn_adj_matrix(self, S: np.ndarray, k, sigma=1.0):
        N = len(S)
        A = np.zeros((N,N))

        for i in range(N):
            dist_index = zip(S[i],range(N))
            dist_index_s = sorted(dist_index, key = lambda x:x[0])
            print(dist_index_s)
            n_index = [dist_index_s[m][1] for m in range(k+1)]
            for ix in n_index:
                A[i][ix] = np.exp(-S[i][ix]/2/sigma/sigma)
                A[ix][i] = A[i][ix]
        return A

    def compute_lapcian(self, adj_m: np.ndarray):
        d_m = np.sum(adj_m,axis=1)
        l_m = np.diag(d_m) - adj_m
        d_factor_m = np.diag(1.0/(d_m)**(0.5))
        return np.dot(np.dot(d_factor_m,l_m),d_factor_m)

    



    def fit(self, data):
        print("type of data is: ", type(data), " shape of data is: ", data.shape)
        # 作业 spectral clustering
        # 屏蔽开始
        s_m = self.compute_sim_matrix(data)
        print("shape of s m: {}".format(s_m.shape))
        adj_m = self.compute_knn_adj_matrix(s_m, self.neighbor_)
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



        # 屏蔽结束

    def predict(self, p_datas):
        result = []
        # 作业2
        # 屏蔽开始
        result = self.km_.predict(p_datas)


        # 屏蔽结束
        return result

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k_means = Spectral_Clustering(n_clusters=2, k=1)
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

