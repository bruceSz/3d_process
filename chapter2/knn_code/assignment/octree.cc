
#include <algorithm>
#include <iostream>
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <limits>

using namespace std;
using namespace Eigen;

using DataM =  Matrix<double, Dynamic, 3>;
struct Point {
        float x;
        float y;
        float z;
    };
//using DataM = vector<Point>;


struct  RadiusNNResultSet {

    public:

    RadiusNNResultSet():radius_(0.0), count_(0), 
        worst_dist_(std::numeric_limits<float>::max()), dist_index_list_({}), comparison_counter_(0) {

        }

    int size() const {
        return count_;
    }


    float worstDist() const {
        return worst_dist_;
    }

    vector<int> top(int n) {
        if (dist_index_list_.size() == 0) {
            return vector<int>();
        }
        auto tmpL = dist_index_list_;
        sort(tmpL.begin(), tmpL.end(),
         [](const std::pair<float,int> p1, const std::pair<float,int> p2 ){ 
             if (p1.first <= p1.first) {
                 return true;
             }
             return false;

         });
         vector<int> res;
         transform(tmpL.cbegin(), tmpL.cend(), std::back_inserter(res) ,[](const std::pair<float, index>p ) {
             return p.second;
         })
         return res;
    }

    void add_points(float dist, int idx) {
        comparison_counter_ += 1;
        if (dist > radius_) {
            return;
        }
        count_ +=1;
        dist_index_list_.push_back(make_pair(dist, idx));
    }

    float radius_;
    int count_;
    float worst_dist_ ;
    vector<std::pair<float, int>> dist_index_list_;
    int comparison_counter_ = 0;

};




class Octant {
    public:

    Octant(Vector3d& center, float extent, vector<int>& point_indices, bool is_leaf) {
        children_.resize(8,nullptr);
        extent_ = extent;
        center_ = center;
        point_indices_ = point_indices;
        is_leaf_ = is_leaf;

    }
    vector<shared_ptr<Octant>> children_;
    Vector3d center_;
    float extent_;
    vector<int> point_indices_;
    bool is_leaf_;
};


//# 功能：通过递归的方式构建octree
//# 输入：
//#     root：根节点
//#     db：原始数据
//#     center: 中心
//#     extent: 当前分割区间
//#     point_indices: 点的key
//#     leaf_size: scale
//#     min_extent: 最小分割区间
shared_ptr<Octant>  octree_recursive_build(shared_ptr<Octant> root,
    DataM  db, Vector3d& center, float  extent, vector<int>& point_indices, int leaf_size, 
     float min_extent) {
        
    if (point_indices.size() == 0) {
        return nullptr;
    }
     
    if (!root){
        root = std::make_shared<Octant>(center, extent, point_indices, true);
        

    }
    if (point_indices.size() <= leaf_size || extent < min_extent) {
        root->is_leaf_ = true;
    } else {
        root->is_leaf_ = false;
        vector<vector<int>> children_point_indices(8, vector<int>());
        for(auto idx: point_indices) {
            //auto pt = db[idx];
            int mortan_code = 0;
            if(db(idx,0) > center[0]) {
                mortan_code |= 1;
            }
            if (db(idx,1) > center[1]) {
                mortan_code |= 2;
            }

            if (db(idx,2) > center[2]) {
                mortan_code |= 4;
            }
            children_point_indices[mortan_code].push_back(idx);
        }
        float factor[] = {-0.5, 0.5};
        int idx_list[] = {0,1,2,3,4,5,6,7};
        for(int idx : idx_list) {
            // for each octant
            Vector3d c_;
            c_[0] = center[0]  + factor[(idx & 1)>0] * extent;
            c_[1] = center[1]  + factor[(idx & 2) > 0] * extent;
            c_[2] = center[2] + factor[(idx & 4) > 0] * extent;
            float child_extent = extent / 2.0;
            root->children_[idx] = octree_recursive_build(root->children_[idx], 
                    db, c_, 
                    child_extent, 
                    children_point_indices[idx], leaf_size , min_extent);

        }

    }
    return root;

}


//# 功能：构建octree，即通过调用octree_recursive_build函数实现对外接口
//# 输入：
//#    dp_np: 原始数据
//#    leaf_size：scale
//#    min_extent：最小划分区间

shared_ptr<Octant> octree_construction(DataM & db, int leaf_size, float min_extent) {
    int N = db.rows();
    int dim = db.cols();
    vector<int> indices;
    std::generate(indices.begin(), indices.end(), [n=0]() mutable{return n++;});
    auto minV = db.colwise().minCoeff();
    auto maxV = db.colwise().maxCoeff();
    float extent = (maxV - minV).maxCoeff() * 0.5;
    Vector3d center = db.colwise().mean();
    
    std::shared_ptr<Octant> root;
    root = octree_recursive_build(root, db, center, extent, indices, leaf_size, min_extent );

    return root;
   // db_min_x = 
}




//# 功能：在octree中查找信息
//# 输入：
//#    root: octree
//#    db：原始数据
//#    result_set: 索引结果
//#    query：索引信息

bool octree_radius_search_fast(shared_ptr<Octant> root , DataM&  db, 
      shared_ptr<RadiusNNResultSet> result_set, Vector3d& query) {
        if (!root) {
            return false;
        }
        if (root->is_leaf_  && root->point_indices_.size() > 0) {
            auto leaf_points = db(root->point_indices_,all);
            
        }


}
    
    
    if  root.is_leaf  and len(root.point_indices) > 0:
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        # check whether we can stop search now
        return inside(query, result_set.worstDist(), root)
    
    # current radius ball contains the octant
    if contains(query, result_set.worstDist(), root) and len(root.point_indices) > 0:
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return False

    # search sub octants.
    morton_code = 0
    if query[0] > root.center[0]:
        morton_code = morton_code | 1
    if query[1] > root.center[1]:
        morton_code = morton_code | 2
    if query[2] > root.center[2]:
        morton_code = morton_code | 4
    # current octant is first priority
    if octree_radius_search_fast(root.children[morton_code], db, result_set, query):
        return True
    for c, child in enumerate(root.children):
        if c == morton_code or child is None:
            continue
        if not overlaps(query, result_set.worstDist(), child):
            continue
        
        if octree_radius_search_fast(child,db, result_set, query):
            return True

    # final check
    return inside(query, result_set.worstDist(), root)
    # 屏蔽结束


int main(int argc, char** argv) {
    cout << "octree speed test: " << std::endl;
    cout << "octree speed test done.  " << std::endl; 

}