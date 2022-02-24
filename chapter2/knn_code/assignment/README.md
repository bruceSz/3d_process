# NN-Trees
Python implementation of Binary Search Tree, kd-tree for tree building, kNN search, fixed-radius search.

## 作业说明
作业入口在benchmark.py，可以直接执行。
执行示例输出如下：
```
(3d_process) brucesz@LAPTOP-26R7SA9G:/data/git/3d_process/chapter2/knn_code/assignment$ python benchmark.py 
octree --------------
shape of db:  (3, 124668)
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 124668/124668 [00:04<00:00, 26807.85it/s]
Octree: build 4673.885, knn 0.757, radius 0.382, brute 10.622
kdtree --------------
Kdtree: build 9.139, knn 1.512, radius 0.477, brute 2.956, scipy 3.452
```
输出 ocTree 和 kdtree的 build和search时间
例如上述： octree build后 4673.885表示octree 创建时间，0.757为knn 搜索时间， 0.382为radius_search_fast时间，10.622为bruce 时间