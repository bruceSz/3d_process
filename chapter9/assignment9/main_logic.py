#// /**
#//  * @file main_logic.py
#//  * @author brucesz
#//  * @date 2022/03/20 22:32:58
#//  * @version $Revision$
#//  * @brief
#//  *
#//  **/


def match_between_ds(s_ds, t_ds):
    """ 
        match between ds in descriptor space
    """
    t_st = o3d.geometry.KDTreeFlann(t_ds)
    matches = []
    for i in len(s_ds):
        ft = s_ds[i]
        _, idx, _ = t_st.search_knn_vector_3d(ft, 1)
        matches.append((i, idx[0]))

    return mathces


def cf_icp(s, t):
    """
        Close form solve icp.
    """
    us = s.mean(axis=0)
    ut = t.mean(axis=1)

    s_center = s - us
    t_center = t - ut
    
    U, s, V  = np.linalg.svd(np.dot(s_center, t_center.T), full_matrix=True, compute_uv=True )
    R = np.dot(U,V)
    t = ut - np.dot(R, us)

    res = np.zeros((4,4))
    res[0:3,0:3] = R
    res[0:3,3] = t
    res[3,3] = 1
    return res

def ransac_init(s_kps_pc, s_ds, t_kps_pc, t_ds):
    #s_st = o3d.geometry.KDTreeFlann(s_ds)
    matches = match_between_ds(s_ds, t_ds)

    st_t_pc = o3d.geometry.KDTreeFlann(t_kps_pc)
    
    

