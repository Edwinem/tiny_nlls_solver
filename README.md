# Tiny NLLS Solver


This is a small non-linear least squares solver. I created it cause I found
myself creating an ugly hack version when trying to implement dense visual slam
problems. A good amount of this project derives from the tiny solver in Ceres.


## How to use


Examples:

```cpp

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

## Advanced Usages
todo

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