# 文件功能：实现 GMM 算法

import numpy as np
from numpy import *
import pylab
import random,math

import numpy as np


from common import gen_dataset
from common import color_array
from common import get_default_para
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
#from scipy.stats import multivariate_normal
plt.style.use('seaborn')

class GMM(object):
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        #self.dim = dim
        #centroid
        
    
    # 屏蔽开始
   
    def init_para(self, data):
        n = data.shape[0]
        dim = data.shape[1]
        self.mu = data[np.random.choice(range(n),self.n_clusters)]
        # equal p for z (each cluster)
        self.pi = np.ones(self.n_clusters)/self.n_clusters

        # var here assume to be diag matrix.
        # each dim is orthogonal
        self.var = np.full((self.n_clusters,dim, dim), np.diag(np.full(dim, 0.1)))

        # W_ij refers to possibility of  each data_i belongs to cluster j
        self.W = np.ones([n,self.n_clusters])/self.n_clusters

    

    
    def fit(self, data):
        n = data.shape[0]
        dim = data.shape[1]
        # 作业3
        # 屏蔽开始
        #1. init z(pi),mu,var
        self.init_para(data)

        # 
        for i in range(self.max_iter):
            #print("{}th iter: cov: {} det of dev{}".format(i,self.var, np.linalg.det(self.var)))
            #1. e-step
            # 更新W
            # compute posterior
            for i in range(n):
                # for each cluster
                for j in range(self.n_clusters):
                    if self.var[j] is None:
                        raise RuntimeError("xxx")
                    self.W[i,j]= self.pi[j] * multivariate_normal.pdf(data[i], self.mu[j], np.diag(self.var[j]))
                # update w
                self.W[i,:] = self.W[i,:]/ sum(self.W[i,:])
                    
            # 更新pi
            # update pi.
            self.pi = self.W.sum(axis=0)/self.W.sum()

                

            #2. m-step
            # 更新Mu
            # 2.1 update mu
            for i in range(self.n_clusters):
                #each ith cluster
                for j in range(dim):
                    self.mu[i][j] = np.average( data[:,j] , weights= self.W[:,i])

            # 更新Var
            # 2.2 update var
            for i in range(self.n_clusters):
                for j in range(dim):
                    tmp = np.average((data[:,j] - self.mu[i][j])**2,weights=self.W[:,i])
                    #print("var is: ",tmp)
                    self.var[i][j]  = tmp

         
            # add epsilon to var
            self.var = self.var + np.tile(np.eye(dim) * 0.00001,(self.n_clusters,1)).reshape(-1, dim, dim) # np.zeros(self.n_clusters,dim,dim)0.00001  * np.eye(dim)
    def get_mu(self):
        return self.mu

    def get_var(self):
        return self.var

        # 屏蔽结束
    
    def predict(self, data):
        # 屏蔽开始
        n = data.shape[0]
        res = np.zeros((n,self.n_clusters))
        for i in range(n):
            for j in range(self.n_clusters):
            
               res [i][j]= self.pi[j] * multivariate_normal.pdf(data[i],self.mu[j], np.diag(self.var[j]))

            res[i,:] = res[i,:]/sum(res[i,:])
        pred = np.argmax(res,axis=1)#[ np.argmax(res[i,:]) for i in range(n)]
        return pred
        # 屏蔽结束
# 屏蔽结束

def plot_x(X1, X2, X3):
    # 显示数据
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.show()

def plot_x_with_Class(X1, X2, X3,mu_list, var_list):
    colors = ['b', 'g', 'r']
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)

    assert len(mu_list) == len(var_list)
    print("shape of mu: {}".format(mu_list))
    print("shape of sigma: {}".format(var_list))
    ax = plt.gca()
    for i in range(len(mu_list)):
        plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': colors[i], 'ls': ':'}
        ellipse = Ellipse(mu_list[i], 3 * np.diag(var_list[i])[0], 3 * np.diag(var_list[i])[1], **plot_args)
        ax.add_patch(ellipse)
    #if (Mu_true is not None) & (Var_true is not None):
    #    for i in range(n_clusters):
    #        plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': colors[i], 'alpha': 0.5}
    #        ellipse = Ellipse(Mu_true[i], 3 * Var_true[i][0], 3 * Var_true[i][1], **plot_args)
    #        ax.add_patch(ellipse) 

    plt.show()
# 生成仿真数据
def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    #X = [X1, X2, X3]
    return X

def main():
# 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)
    #X = np.vstack((X_list[0],X_list[1],X_list[2]))
    gmm = GMM(n_clusters=3)
    gmm.fit(X)
    cat = gmm.predict(X)
    #print("cat size is: {} ".format(cat.shape))
    #print(cat)
#   
    plot_x_with_Class(X[0:400,:],X[400:1000,:],X[1000:,:],gmm.get_mu(),gmm.get_var())

    # 初始化


def test_with_datasets():
    da = gen_dataset()
    plot_num = 1
    default_base = get_default_para()
    for i, (data, algo_para) in enumerate(da):
        params = default_base.copy()
        params.update(algo_para)
        
        X, label = data
        print("shape of data {}".format(X.shape))

        gmm = GMM(n_clusters=params["n_clusters"])
        gmm.fit(X)
        pred = gmm.predict(X)
        print(np.unique(pred))
        c_set = color_array()
        colors = []
        for i in pred:
            colors.append(c_set[i])
        plt.subplot(len(da), 1, plot_num)
        plt.scatter(X[:,0],X[:,1],c=colors)
        #plt.show()

        plot_num += 1

    plt.show()

def gaussian(x, mean, cov):
    dim = np.shape(cov)[0]
    print(dim)
    covdet = np.linalg.det(cov + np.eye(dim) * 0.00001)
    covinv = np.linalg.inv(cov + np.eye(dim)* 0.00001)
    xdiff = (x-mean).reshape(1,dim)
    p = 1.0/(np.power(np.power(2*np.pi,dim)*np.abs(covdet),0.5)) * np.exp(-0.5*xdiff.dot(covinv).dot(xdiff.T))[0][0]
    return p

def print_gaussian():
    mu = np.array([0.5, 0.5])
    var = np.array([[1,0],[0,2]])
    x = np.array([2,3])
    print(gaussian(x, mu, var))

    from scipy.stats import multivariate_normal
    print("scipy normal result: ",multivariate_normal.pdf(x,mu,var))

if __name__ == '__main__':
    test_with_datasets()



    

