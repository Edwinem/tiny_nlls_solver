//
// Created by nikolausmitchell on 5/18/19.
//



#include "tiny_solver.h"



   struct MyCostFunctionExample {
     typedef double Scalar;
     enum {
       NUM_RESIDUALS = 2,
       NUM_PARAMETERS = 3,
     };
     bool operator()(const double* parameters,
                     double* residuals,
                     double* jacobian) const {
       int x,y,z=0;
       residuals[0] = x + 2*y + 4*z;
       residuals[1] = y * z;
       if (jacobian) {
         jacobian[0 * 2 + 0] = 1;   // First column (x).
         jacobian[0 * 2 + 1] = 0;

         jacobian[1 * 2 + 0] = 2;   // Second column (y).
         jacobian[1 * 2 + 1] = z;

         jacobian[2 * 2 + 0] = 4;   // Third column (z).
         jacobian[2 * 2 + 1] = y;
       }
       return true;
     }
   };

struct S {
  double operator()(char, int&);
  float operator()(int) { return 1.0;}
};

   struct MyCostFunctionExampleHessian {
     typedef double Scalar;
     enum {
       NUM_RESIDUALS = 2,
       NUM_PARAMETERS = 3,
     };
     bool operator()(const double* parameters,
                     double* residual,
                     double* gradient,
                     double* hessian) const {

       residual[0]=5;
       if (hessian) {
       }
       return true;
     }
   };

int main(){


  ts::TinySolver<MyCostFunctionExample> s;

  MyCostFunctionExampleHessian u;

  MyCostFunctionExample e;

  Eigen::Matrix<double, 3, 1> data;

  std::result_of<MyCostFunctionExample(const double* parameters,
                   double* residuals,
                   double* jacobian)>::type d = true; // d has type double

//  std::result_of<MyCostFunctionExample(const double* parameters,       double* gradient,
//                       double* hessian,double* cost)>::type d=true;

  s.options.minimizer=s.GAUSSNEWTON;

  s.Solve(e,&data);

    return 0;
}