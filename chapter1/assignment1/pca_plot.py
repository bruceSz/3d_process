import utils

def plot_after_pca():
    p_dir = "../dataset/modelnet40_normal_resampled"
    m_ = utils.get_pcd_cls(p_dir)
    da = utils.load_datas(m_)
    print(da.keys())
   # for target in da.keys():
    for target in ["airplane"]:
        #arget="airplane"
        # target = "bottle"
        print(da[target].shape)
        air_plane_pcd = da[target][:,0:3]
        print(air_plane_pcd)
        #print(air_plane_pcd.shape)
        res,_,_ = utils.pca(air_plane_pcd, number_pc = 2)
        res2 = utils.plot_pcd_from_pts(res)
    #print(air_plane_pcd[:10,:])
    #print(res[:10,:])

if __name__ == "__main__":
    plot_after_pca()