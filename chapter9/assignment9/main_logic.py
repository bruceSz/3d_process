#// /**
#//  * @file main_logic.py
#//  * @author brucesz
#//  * @date 2022/03/20 22:32:58
#//  * @version $Revision$
#//  * @brief
#//  *
#//  **/

from scipy.spatial.distance import pdist
import open3d as o3d
import numpy as np
import copy


def match_between_ds(s_ds, t_ds):
    """ 
        match between ds in descriptor space
    """
    print("shape of target descriptor: ", t_ds.shape)
    print("shape of source descriptor: ", s_ds.shape)
    
    t_st = o3d.geometry.KDTreeFlann(t_ds.T)
    matches = []
    print("type of s_ds", type(s_ds))
    for i in range(len(s_ds)):
        ft = s_ds[i]
        #print("shape of ft: ", ft.shape)
        _, idx, _ = t_st.search_knn_vector_xd(ft, 1)
        matches.append((i, idx[0]))

    return matches


def cf_icp(s, t):
    """
        Close form solve icp.
    """
    us = s.mean(axis=0)
    ut = t.mean(axis=0)

    s_center = s - us
    t_center = t - ut
    #print("shape of us: ", s_center.shape)
    #print("shape of ut: ", t_center.shape)

    
    U, s, V  = np.linalg.svd(np.dot(t_center.T, s_center), full_matrices=True, compute_uv=True)
    #print("shape of U", U.shape)
    #print("shape of V", V.shape)
    R = np.dot(U,V)
    t = ut - np.dot(R, us)

    res = np.zeros((4,4))
    res[0:3,0:3] = R
    res[0:3,3] = t
    res[3,3] = 1
    return res

def is_valid_match_norm_cosine(pcd_s, pcd_t, s_idx, t_idx, norm_st_angle_cos_thresh):
    
    norms_s = pcd_s.normals[s_idx]
    norms_t = pcd_t.normals[t_idx]

    norm_cos = (norms_s * norms_t).sum(axis=1)
    # there is a norm_cos deviation smaller than thresh.
    not_valid = np.any(norm_cos < norm_st_angle_cos_thresh)
    return not not_valid

def is_valid_match_edge_lenth(pcd_s, pcd_t, s_idx, t_idx, thres):

    points_s = np.asarray(pcd_s.points)[s_idx]
    points_t = np.asarray(pcd_t.points)[t_idx]

    dist_s = pdist(points_s)
    dist_t = pdist(points_t)

    diff_m = np.absolute(dist_s - dist_t) / np.minimum(dist_s, dist_t)

    is_valid = np.all(diff_m < thres  )
    return is_valid


def is_valid_near(pcd_s, pcd_t, s_idx, t_idx, thresh):

    #print("max of s_idx:",np.max(s_idx))
    #print("max of t_idx:",np.max(t_idx))
    #print("shape of pcd_s: ", len(pcd_s.points))
    #print("shape of pcd_t: ", len(pcd_t.points))
    

    points_s = np.asarray(pcd_s.points)[s_idx]
    points_t = np.asarray(pcd_t.points)[t_idx]

    assert(points_s.shape == points_t.shape)
    T = cf_icp(points_s, points_t)
    R,t = T[0:3, 0:3], T[0:3,3]

    diff = np.linalg.norm(
        points_t - np.dot(points_s, R.T) - t,
        axis=1
    )
    #print("diff is: ", diff)
    is_valid = np.all(diff < thresh)

    return is_valid, T



def shall_terminate(curr, prev):
    gain = curr.fitness/prev.fitness -1
    return gain < 0.005

max_distance= 5
max_iter= 20000

def exact_match(pcd_s, pcd_t, st_t, T ):

    """
    perform exact match from point cloud source to point cloud target.
    Parameters
    -------

    Returns
    -------
    
    """

    N  = len(pcd_s.points)

    curr = prev = o3d.pipelines.registration.evaluate_registration(
        pcd_s, pcd_t, max_distance, T
    )

    best_T = T
    for icp_iter in range(max_iter):
        pcd_s_curr = copy.deepcopy(pcd_s)
        pcd_s_curr = pcd_s_curr.transform(T)

        matches = []
        for i in range(N):
            query = np.array(pcd_s_curr.points)[i]
            _, idx, dis = st_t.search_knn_vector_3d(query, 1)

            if dis[0] <= max_distance:
                matches.append((i, idx[0]))
            
        matches = np.array(matches)

        if len(matches) >= 4:
           #print("do icp last time")
            A = np.asarray(pcd_s.points)[matches[:,0]]
            B = np.asarray(pcd_t.points)[matches[:,1]]
            T = cf_icp(A, B)
            #print("update T", T)

            curr = o3d.pipelines.registration.evaluate_registration(pcd_s, pcd_t, max_distance, T)
            print("icp iter: ", icp_iter , "fitness" , curr.fitness)
            if shall_terminate(curr, prev):
                print("got best ", prev.fitness, " curr fitness:", curr.fitness)
                break
            if curr.fitness > prev.fitness:
                print("update result with new fitness", curr.fitness)
                prev = curr
                best_T = T

    return curr,best_T

def get_ransac_guess(s_kps_pc, t_kps_pc, s_ds, t_ds):
    matches = match_between_ds(s_ds, t_ds)
    
    matches = np.asarray(matches)
    
    st_t_pc = o3d.geometry.KDTreeFlann(t_kps_pc)
    

    idx_matches = np.arange(len(matches))
    iter = 0
    best_T = None
    best_fit = -1
    for iter in range(10):
        print("ransac iter : ", iter)
        while True:
            s_idx = np.random.choice(idx_matches, 4)
            #print(s_idx)
            s_idx = matches[s_idx,0]
            #print(s_idx)
            t_idx = matches[s_idx,1]
            #print(t_idx)
            is_valid,T = is_valid_near(s_kps_pc, t_kps_pc, s_idx, t_idx, max_distance)
            if is_valid:
                found_s_idx  = s_idx
                print("found !!!!!")
                break
            iter += 1
        
        curr = o3d.pipelines.registration.evaluate_registration(s_kps_pc, t_kps_pc, max_distance, T)
        if curr.fitness > best_fit:
            best_fit = curr.fitness
            best_T = T
            print("ransacn update T with fitness", best_fit)

    return best_T


def ransac_match(s_kps_pc, t_kps_pc, s_ds, t_ds):
    #s_st = o3d.geometry.KDTreeFlann(s_ds)
    # 在描述子空间中， 找匹配。
    
    matches = match_between_ds(s_ds, t_ds)
    assert(len(s_kps_pc.points) == len(s_ds))
    assert(len(t_kps_pc.points) == len(t_ds))
    matches = np.asarray(matches)
    print("shape of mathces: ", matches.shape)
    print("max index: ",np.max(matches, axis=0))

    st_t_pc = o3d.geometry.KDTreeFlann(t_kps_pc)
    

    idx_matches = np.arange(len(matches))
    #proposal_generator = (
    #    matches[np.random.choice(idx_matches, ransac_params.num_samples, replace=False)] for _ in iter(int, 1)
    #)
    found_s_idx = None
    T = None
    iter = 0
    while True:
        print("ransac : ", iter)
        s_idx = np.random.choice(idx_matches, 4)
        #print(s_idx)
        s_idx = matches[s_idx,0]
        #print(s_idx)
        t_idx = matches[s_idx,1]
        #print(t_idx)
        is_valid,T = is_valid_near(s_kps_pc, t_kps_pc, s_idx, t_idx, max_distance)
        if is_valid:
            found_s_idx  = s_idx
            print("found !!!!!")
            break
        iter += 1
    
    
    print("ransac match are: ", found_s_idx)
    print("init T : ", T)
    best_curr = exact_match(s_kps_pc, t_kps_pc, st_t_pc, T)


    for item in range(1000):
        s_idx = np.random.choice(idx_matches,4)
        t_idx = matches[s_idx,1]
        points_s = np.asarray(s_kps_pc.points)[s_idx]
        points_t = np.asarray(t_kps_pc.points)[t_idx]
        assert(points_s.shape == points_t.shape)
        T = cf_icp(points_s, points_t)
        result  = exact_match(s_kps_pc, t_kps_pc, st_t_pc, T)
        
        best_curr = best_curr if best_curr.fitness > result.fitness else result
        
    return T

    #for p in proposal_generator():

