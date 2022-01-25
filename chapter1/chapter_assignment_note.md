## scripts

1. pca_plot.py
2. downsample_voxel.py
3. normal_estimation.py
4. utils.py (helper.py file)

## results:
1. 2_pca_results:, 可以使用pca_plot.py 生成
  这个目录保存了对每个modelnet40中类别选中的数据进行pca处理后，选择前两个（最大特征值和次大特征值对应特征向量），reconstruction 后可视化，截图结果。

2. normal_estimation 结果 可以使用normal_estimation.py生成
 这里只对aireplane类别一个对象进行了normal的估计，
   * airplane_with_existing_normal.png是对课程提供的数据集自带的normal进行可视化后截图
   * airplane_with_computed_normal.png 是对数据点云进行normal估计后，可视化后截图结果
   * 两者对比用

3. 下采样 可以使用downsample_voxel.py生成
  * 结果存于down目录
  * 针对每个modelnet40中类型一个点云，运行基于center和基于random的两个下采样方法，可视化截图后见center.png和random.png后缀以作区分。

