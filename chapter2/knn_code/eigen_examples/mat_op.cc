#include <iostream>
#include <Eigen/Dense>
//#include <Eigen/Core>
using namespace Eigen;
using namespace std;
using Eigen::all;

int mul() {
    
    Matrix2d mat;  mat << 1, 2,         3, 4;  
    Vector2d u(-1,1), v(2,0);  
    std::cout << "Here is mat*mat:\n" << mat*mat << std::endl;  
    std::cout << "Here is mat*u:\n" << mat*u << std::endl;  
    std::cout << "Here is u^T*mat:\n" << u.transpose()*mat << std::endl; 
    std::cout << "Here is u^T*v:\n" << u.transpose()*v << std::endl;
    std::cout << "Here is u*v^T:\n" << u*v.transpose() << std::endl;  
    std::cout << "Let's multiply mat by itself" << std::endl;  
    mat = mat*mat;  std::cout << "Now mat is mat:\n" << mat << std::endl;
    return 0;

}

int dot_cross() {
    
    Matrix2d mat;  mat << 1, 2,         3, 4;  
    Vector2d u(-1,1), v(2,0);  
    std::cout << "Here is mat*mat:\n" << mat*mat << std::endl;  
    std::cout << "Here is mat*u:\n" << mat*u << std::endl;  
    std::cout << "Here is u^T*mat:\n" << u.transpose()*mat << std::endl; 
    std::cout << "Here is u^T*v:\n" << u.transpose()*v << std::endl;
    std::cout << "Here is u*v^T:\n" << u*v.transpose() << std::endl;  
    std::cout << "Let's multiply mat by itself" << std::endl;  
    mat = mat*mat;  std::cout << "Now mat is mat:\n" << mat << std::endl;
    return 0;

}
int add_sub()
{  
    Matrix2d a;  a << 1, 2,       3, 4;  
    MatrixXd b(2,2); 
    b << 2, 3,       1, 4; 
    cout << "a is:\n " << a << endl;
    cout << "b is \n: " << b << endl;
    std::cout << "a + b =\n" << a + b << std::endl;
    std::cout << "a - b =\n" << a - b << std::endl; 
    std::cout << "Doing a += b;" << std::endl; 
    a += b;  
    std::cout << "Now a =\n" << a << std::endl; 
    Vector3d v(1,2,3);  
    Vector3d w(1,0,0);  
    std::cout << "-v + w - v =\n" << -v + w - v << std::endl;

    mul();
}

int vec() {
    Vector2d a(5.0, 6.0);
    Vector3d b(5.0, 6.0, 7.0);
    //Matrix2d c(5.0,6.0);
    cout << typeid(b).name() << endl;
    return 0;
}

int slice() {
    ArrayXi ind(5); ind<<4,2,5,5,3;
    MatrixXi A = MatrixXi::Random(4,6);
    cout << "Initial matrix A:\n" << A << "\n\n";
    cout << "A(all,ind-1):\n" << A(Eigen::all,ind-1) << "\n\n";
    return 1;
}
int main() {
    dot_cross();
    vec();
    slice();
}