// 代码片段1：运行时设置矩阵的维度
#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;

int runtime_dim()
{
cout << "runtime dim begin" << endl;
  
 
    MatrixXd m =  MatrixXd::Random(3,3);
  cout << "origin m: " << m << endl;
  MatrixXd m2 = (m + MatrixXd::Constant(3,3,3)) ;
  cout << "m =" << endl << m2 << endl;
  VectorXd v(3);
  v << 1, 2, 3;
  cout << "m * v =" << endl << m * v << endl;
  cout << "runtime dim end" << endl;
  return 0;
}

// 代码片段2：编译时确定矩阵的维度
/*int compile_dim()
{
    return 1;
  Matrix3d m = Matrix3d::Random();
  return 1;
  m = (m + Matrix3d::Constant(1.2)) * 50;
  cout << "m =" << endl << m << endl;
  Vector3d v(1,2,3);
  
  cout << "m * v =" << endl << m * v << endl;
}
*/


int main() {
    
    cout << "mat and vec dim done." << endl;
    runtime_dim();
    //compile_dim();
    cout << "mat and vec dim done." << endl;
    return 0;
}
