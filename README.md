# Tiny NLLS Solver


This is a small non-linear least squares solver. I created it cause I found
myself creating an ugly hack version when trying to implement dense visual slam
problems. A good amount of this project derives from the tiny solver in Ceres.

## Requirements

Required:
* Eigen

Optional:
* gtest
* Open3D
* OpenCV

## How to use

The only actual file you need is [src/tiny_solver.h](src/tiny_solver.h). You can
copy this into your project.

Example on how to use the solver:
```cpp
#include <src/tiny_solver.h>


//Define cost function
class ExampleCostFunction {
 public:
  typedef double Scalar;
  enum {
    // Can also be Eigen::Dynamic.
    NUM_RESIDUALS = 2,
    NUM_PARAMETERS = 3,
  };
  bool operator()(const double* parameters,
                  double* residuals,
                  double* jacobian) const {
      Scalar x = parameters[0];
      Scalar y = parameters[1];
      Scalar z = parameters[2];
      residuals[0] = x + 2 * y + 4 * z;
      residuals[1] = y * z;
      if (jacobian) {
        jacobian[0 * 2 + 0] = 1;
        jacobian[0 * 2 + 1] = 0;
    
        jacobian[1 * 2 + 0] = 2;
        jacobian[1 * 2 + 1] = z;
    
        jacobian[2 * 2 + 0] = 4;
        jacobian[2 * 2 + 1] = y;
      }
      return true;
  }
};

//Create the tiny solver
ts::TinySolver<ExampleCostFunction> solver;
ExampleCostFunction cost_functor;

Vec3 initial_guess(0.76026643, -30.01799744, 0.55192142);
solver.Solve(cost_functor,&initial_guess);
Vec3 answer = initial_guess;

std::cout << answer << std::endl;
```

The other files in **src** are optional. They can help if you want to wrap existing
Ceres cost functions or use Ceres's autodiff functionality.

For less contrived examples I recommend perusing **/examples** specifically
[triangulation_example.cpp](examples/triangulation_example.cpp) and
 [triangulation_example_advanced.cpp](examples/triangulation_example_advanced.cpp).
 They both show the same method to minimize the position of a 3D point viewed
 from multiple cameras. The advanced version shows how to use the hessian based
 cost function.


## Advanced Usages

### Hessian based Cost function
The solver supports another type of cost function which allows you to build the
hessian and gradient matrices directly. The functor should look like the
 following
```cpp

class ExampleHessianCostFunction {
 public:
  typedef double Scalar;
  enum {
    NUM_RESIDUALS = 2,
    NUM_PARAMETERS = 3,
  };
  bool operator()(const double* parameters,
                  double* residual,
                  double* gradient,
                  double* hessian) const {
    // do something
    return true;
  }
};
```

As you are building the Normal Equations you can directly modify them to suit
your needs. Possible things you can do:
* Custom Preconditioner. You can modify the hessian for better numerical
stability.
* Custom Accumulation methods to build the matrices faster. This could involve 
threading or custom SIMD like in [DVO](https://github.com/tum-vision/dvo/blob/bd21a70ce76d882a354de7b89d2429f974b8814c/dvo_core/include/dvo/core/math_sse.h#L48).
* Weighting functions. You can apply huber norms or other weighting schemes.

### Custom parameterizations
By default the solver does the default additive update
```cpp
 // dx is the solved for update and x_prev is your previous value
 x_new=x_prev+dx
```

You can create your own parameterization function to overwrite this functionality.
 An example for this can be seen below. Here we are optimizing a quaternion
 which has the requirement of norm being one. (Note this is a very naive way to
 parameterize quaternions)
```cpp

template<typename Scalar>
struct QuaternionNormedParameterization {

  void operator()(
      const Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> x_prev,
      const Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> dx,
      Eigen::Ref<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> x_new) {
    Eigen::Vector<Scalar,4,1> q; //quaternion
    q = x_prev + dx;
    q.normalize();
    x_new=q;
  }
};

struct CostFunction{
 //some cost function
}

using LinearSolver=Eigen::LDLT<Eigen::Matrix<typename CostFunction::Scalar,
                               CostFunction::NUM_PARAMETERS,CostFunction::NUM_PARAMETERS> >
//Instantiate Solver like so

ts::TinySolver<CostFunction,LinearSolver,
QuaternionNormedParameterization<typename CostFunction::Scalar>> solver;

```


## Using this directory
This directory provides unit testing and advanced examples on how to use the tiny_solver.

Unit tests can be turned on by enabling the **WITH_TESTING** CMake option which
will also download gtest.

Some examples require additional libraries like OpenCV and Open3D. They are by 
default turned **OFF** and need to be turned on with another CMake option.
Typically something like **WITH_XXXX** where **XXXX** is the desired library
name(e.g OPENCV).

## Possible bugs

- If the cost function struct/class has both the hessian and the standard cost
 function then it will probably break.
 
 **Don't do this**
 ```cpp
 
    struct MyCostFunctionExample {
      typedef double Scalar;
      enum {
        NUM_RESIDUALS = 2,
        NUM_PARAMETERS = 3,
      };
      bool operator()(const double* parameters,
                      double* residuals,
                      double* jacobian) const {
        // do something
        return true;
      }
      
      bool operator()(const double* parameters,
                      double* residual,
                      double* gradient,
                      double* hessian) const {
        // do something
        return true;
      }
    };
 
 ```
 


## To Do


- [ ] Maybe add ability to add weights. Currently user can do it manually in the
 cost function
- [X] Add the ability for the cost function to create the pseudo Hessian + grad
 directly. Many of Direct slam methods such as DSO do this with SIMD. It could
 also allow for custom scaling values.
- [X] Gauss newton
- [ ] Dogleg
- [ ] Add increase in cost as a failure to LM
- [ ] Benchmarks between this and ceres
- [ ] Optional printing of status
- [ ] Parameterizations needs to be a bit cleaner. Currently it is a functor,
however, it is instantiated temporarily in the Solve function call, and thus
can't be used from the outside. Should either be passed with the Solve call or
maybe as a std::function


## Credits

Ceres(http://ceres-solver.org/index.html)

The base of this tiny solver comes from the tiny solver in Ceres. Also I took
their cost functions design and autodiff to make certain things easier.