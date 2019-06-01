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
    return EvaluateResidualsAndJacobians(parameters, residuals, jacobian);
  }
};

//Create the tiny solver

ts::TinySolver<ExampleCostFunction> solver;

ExampleCostFunction cost_functor;

Vec3 initial_guess(0.76026643, -30.01799744, 0.55192142);

solver.Solve(cost_functor,&initial_guess);

Vec3 answer=initial_guess;

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
- [ ] Add increase in cost function as a failure to LM
- [ ] Benchmarks between this and ceres
- [ ] Parameterizations needs to be a bit cleaner. Currently it is a functor,
however, it is instantiated temporarily in the Solve function call, and thus
can't be used from the outside. Should either be passed with the Solve call or
maybe as a std::function


## Credits

Ceres(http://ceres-solver.org/index.html)

The base of this tiny solver comes from the tiny solver in Ceres. Also I took
their cost functions design and autodiff to make certain things easier.