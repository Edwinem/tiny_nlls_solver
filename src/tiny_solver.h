// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2017 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: mierle@gmail.com (Keir Mierle)
//
// WARNING WARNING WARNING
// WARNING WARNING WARNING  Tiny solver is experimental and will change.
// WARNING WARNING WARNING
//
// A tiny least squares solver using Levenberg-Marquardt, intended for solving
// small dense problems with low latency and low overhead. The implementation
// takes care to do all allocation up front, so that no memory is allocated
// during solving. This is especially useful when solving many similar problems;
// for example, inverse pixel distortion for every pixel on a grid.
//
// Note: This code has no dependencies beyond Eigen, including on other parts of
// Ceres, so it is possible to take this file alone and put it in another
// project without the rest of Ceres.
//
// Algorithm based off of:
//
// [1] K. Madsen, H. Nielsen, O. Tingleoff.
//     Methods for Non-linear Least Squares Problems.
//     http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/3215/pdf/imm3215.pdf

#pragma once

#include <cassert>
#include <cmath>

#include <Eigen/Cholesky> //for default LLDT

#include <type_traits>

namespace ts {

// To use tiny solver, create a class or struct that allows computing the cost
// function (described below). This is similar to a ceres::CostFunction, but is
// different to enable statically allocating all memory for the solver
// (specifically, enum sizes). Key parts are the Scalar typedef, the enums to
// describe problem sizes (needed to remove all heap allocations), and the
// operator() overload to evaluate the cost and (optionally) jacobians.
//
//   struct TinySolverCostFunctionTraits {
//     typedef double Scalar;
//     enum {
//       NUM_RESIDUALS = <int> OR Eigen::Dynamic,
//       NUM_PARAMETERS = <int> OR Eigen::Dynamic,
//     };
//     bool operator()(const double* parameters,
//                     double* residuals,
//                     double* jacobian) const;
//
//     int NumResiduals() const;  -- Needed if NUM_RESIDUALS == Eigen::Dynamic.
//     int NumParameters() const; -- Needed if NUM_PARAMETERS == Eigen::Dynamic.
//   };
//
// For operator(), the size of the objects is:
//
//   double* parameters -- NUM_PARAMETERS or NumParameters()
//   double* residuals  -- NUM_RESIDUALS or NumResiduals()
//   double* jacobian   -- NUM_RESIDUALS * NUM_PARAMETERS in column-major format
//                         (Eigen's default); or NULL if no jacobian requested.
//
// An example (fully statically sized):
//
//   struct MyCostFunctionExample {
//     typedef double Scalar;
//     enum {
//       NUM_RESIDUALS = 2,
//       NUM_PARAMETERS = 3,
//     };
//     bool operator()(const double* parameters,
//                     double* residuals,
//                     double* jacobian) const {
//       residuals[0] = x + 2*y + 4*z;
//       residuals[1] = y * z;
//       if (jacobian) {
//         jacobian[0 * 2 + 0] = 1;   // First column (x).
//         jacobian[0 * 2 + 1] = 0;
//
//         jacobian[1 * 2 + 0] = 2;   // Second column (y).
//         jacobian[1 * 2 + 1] = z;
//
//         jacobian[2 * 2 + 0] = 4;   // Third column (z).
//         jacobian[2 * 2 + 1] = y;
//       }
//       return true;
//     }
//   };
//
// The solver supports either statically or dynamically sized cost
// functions. If the number of residuals is dynamic then the CostFunction
// must define:
//
//   int NumResiduals() const;
//
// If the number of parameters is dynamic then the CostFunction must
// define:
//
//   int NumParameters() const;
//
// Custom parameterizations:
//
// In order to support operations such as unit quaternion updates you can
// override the update parameterization by defining your own function
//
// template<typename Scalar>
// struct CustomExtraScalingParameterization {
//  void operator()(
//  const Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> x_prev,
//  const Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> dx,
//  Eigen::Ref<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> x_new)
//  {
//    x_new = x_prev + 3*dx;
//  }
// };
//
// To change just change the template parameter ParameterizationFunction.
// By default it is set to the normal addition operation


// In addition to the standard cost function above this version supports
// directly building the gradient(j*r) and the hessian(jtj).
// Some advanced usages this allows for:
// 1) Custom accumulator classes to build Hessian matrix a lot faster( see the
// dense visual slam projects from TUM)
// 2) Custom scaling of certain parts of the hessian matrix
//
//   struct MyCostFunctionExampleHessian {
//     typedef double Scalar;
//     enum {
//       NUM_RESIDUALS = 2,
//       NUM_PARAMETERS = 3,
//     };
//     bool operator()(const double* parameters,
//                     double* residuals,
//                     double* gradient,
//                     double* hessian) const {
//       residuals[0] = x + 2*y + 4*z;
//       residuals[1] = y * z;
//
//       if (gradient && hessian) {
//
//       Eigen::Matrix<Scalar, NUM_RESIDUALS, NUM_PARAMETERS> jac;
//       //compute Jacobian
//       Eigen::Map<Eigen::Matrix<Scalar, NUM_RESIDUALS, 1>> error(residuals);
//       Eigen::Map<Eigen::Matrix<Scalar, NUM_PARAMETERS, 1>> grad(gradient);
//       Eigen::Map< Eigen::Matrix<Scalar, NUM_PARAMETERS, NUM_PARAMETERS>>
//       jtj(hessian);
//       jtj=jac.transpose()*jac;
//       grad=jac.transpose()*-error;
//
//       }
//       return true;
//     }
//   };



// The standard way to add the delta updates to the parameters. Through
// simple addition. x_new= x_prev+dx
template<typename Scalar>
struct DefaultAdditionParameterization {

  void operator()(
      const Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> x_prev,
      const Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> dx,
      Eigen::Ref<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> x_new) {
    x_new = x_prev + dx;
  }
};

template<typename CostFunction,
    typename LinearSolver = Eigen::LDLT<
        Eigen::Matrix<typename CostFunction::Scalar,
                      CostFunction::NUM_PARAMETERS,
                      CostFunction::NUM_PARAMETERS> >,
    typename ParameterizationFunction =
    DefaultAdditionParameterization<typename CostFunction::Scalar> >
class TinySolver {
 public:
  // This class needs to have an Eigen aligned operator new as it contains
  // fixed-size Eigen types.
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum {
    NUM_RESIDUALS = CostFunction::NUM_RESIDUALS,
    NUM_PARAMETERS = CostFunction::NUM_PARAMETERS
  };
  typedef typename CostFunction::Scalar Scalar;
  typedef typename Eigen::Matrix<Scalar, NUM_PARAMETERS, 1> Parameters;

  enum Status {
    GRADIENT_TOO_SMALL,            // eps > max(J'*f(x))
    RELATIVE_STEP_SIZE_TOO_SMALL,  // eps > ||dx|| / (||x|| + eps)
    COST_TOO_SMALL,                // eps > ||f(x)||^2 / 2
    HIT_MAX_ITERATIONS,
    COST_FUNCTION_FAIL,
    COST_INCREASED

  };

  enum MinimizerMethod {
    LM, //levenberg marquadt
    DOGLEG, //powell dogleg
    GAUSSNEWTON
  };

  struct Options {
    Scalar gradient_tolerance = 1e-10;  // eps > max(J'*f(x))
    Scalar parameter_tolerance = 1e-8;  // eps > ||dx|| / ||x||
    Scalar cost_threshold =             // eps > ||f(x)||
        std::numeric_limits<Scalar>::epsilon();
    Scalar initial_trust_region_radius = 1e4;
    int max_num_iterations = 50;
    MinimizerMethod minimizer;
  };

  struct Summary {
    Scalar initial_cost = -1;       // 1/2 ||f(x)||^2
    Scalar final_cost = -1;         // 1/2 ||f(x)||^2
    Scalar gradient_max_norm = -1;  // max(J'f(x))
    int iterations = -1;
    Status status = HIT_MAX_ITERATIONS;
  };

  bool Update(CostFunction &function, const Parameters &x,
              bool only_compute_cost = false) {

    //Call either the standard jacobian+ error cost function and build
    // jtj and g_ , or call the hessian version
    return UpdateCostFunction(function, x, only_compute_cost);

  }

  const Summary &Solve(CostFunction &function, Parameters *x_and_min) {
    ParameterizationFunction parameterization_function;

    Initialize<NUM_RESIDUALS, NUM_PARAMETERS>(function);
    assert(x_and_min);
    Parameters &x = *x_and_min;
    summary = Summary();
    summary.iterations = 0;

    // TODO(sameeragarwal): Deal with failure here.
    bool suc = Update(function, x);
    if (!suc) {
      summary.status = COST_FUNCTION_FAIL;
      return summary;
    }
    summary.initial_cost = cost_;
    summary.final_cost = cost_;
    Scalar prev_cost = cost_;

    if (summary.gradient_max_norm < options.gradient_tolerance) {
      summary.status = GRADIENT_TOO_SMALL;
      return summary;
    }

    if (cost_ < options.cost_threshold) {
      summary.status = COST_TOO_SMALL;
      return summary;
    }

    switch (options.minimizer) {

      case GAUSSNEWTON: {

        for (summary.iterations = 1;
             summary.iterations < options.max_num_iterations;
             summary.iterations++) {

          linear_solver_.compute(jtj_);
          dx_ = linear_solver_.solve(g_);

          const Scalar parameter_tolerance =
              options.parameter_tolerance *
                  (x.norm() + options.parameter_tolerance);
          if (dx_.norm() < parameter_tolerance) {
            summary.status = RELATIVE_STEP_SIZE_TOO_SMALL;
            break;
          }
          //By default just does x_new_ = x + dx_;
          parameterization_function(x, dx_, x_new_);

          suc = Update(function, x_new_);

          if (!suc) {
            summary.status = COST_FUNCTION_FAIL;
            break;
          }
          //Update made it worse so stop iterations
          if (cost_ > prev_cost) {
            summary.status = COST_INCREASED;
            cost_=prev_cost; //ensures final cost is last good one
            break;
          }
          prev_cost = cost_;
          x = x_new_;
          if (summary.gradient_max_norm < options.gradient_tolerance) {
            summary.status = GRADIENT_TOO_SMALL;
            break;
          }

          if (cost_ < options.cost_threshold) {
            summary.status = COST_TOO_SMALL;
            break;
          }

        }

        break;
      }

      case DOGLEG: {
        Scalar delta_k = 2.0; // trust region radius
        const Scalar delta_max = 8.0; // max trust region radius
        const Scalar eta = 0.125; // min reduction ratio allowed (0<eta<0.25)
        break;
      }

      default: //default is Levengberg Marquadt
      case LM: {
        Scalar u = 1.0 / options.initial_trust_region_radius;
        Scalar v = 2;

        for (summary.iterations = 1;
             summary.iterations < options.max_num_iterations;
             summary.iterations++) {
          jtj_regularized_ = jtj_;
          const Scalar min_diagonal = 1e-6;
          const Scalar max_diagonal = 1e32;
          for (int i = 0; i < lm_diagonal_.rows(); ++i) {
            lm_diagonal_[i] = std::sqrt(
                u * std::min(std::max(jtj_(i, i), min_diagonal), max_diagonal));
            jtj_regularized_(i, i) += lm_diagonal_[i] * lm_diagonal_[i];
          }

          // TODO(sameeragarwal): Check for failure and deal with it.
          linear_solver_.compute(jtj_regularized_);
          lm_step_ = linear_solver_.solve(g_);
          dx_ = jacobi_scaling_.asDiagonal() * lm_step_;

          // Adding parameter_tolerance to x.norm() ensures that this
          // works if x is near zero.
          const Scalar parameter_tolerance =
              options.parameter_tolerance *
                  (x.norm() + options.parameter_tolerance);
          if (dx_.norm() < parameter_tolerance) {
            summary.status = RELATIVE_STEP_SIZE_TOO_SMALL;
            break;
          }
          //By default just does x_new_ = x + dx_;
          parameterization_function(x, dx_, x_new_);


          // Compute costs with new parameters
          suc = Update(function, x_new_, true);
          if (!suc) {
            summary.status = COST_FUNCTION_FAIL;
            break;
          }

          const Scalar cost_change = (2 * cost_ - f_x_new_.squaredNorm());

          // TODO(sameeragarwal): Better more numerically stable evaluation.
          const Scalar
              model_cost_change = lm_step_.dot(2 * g_ - jtj_ * lm_step_);

          // rho is the ratio of the actual reduction in error to the reduction
          // in error that would be obtained if the problem was linear. See [1]
          // for details.
          Scalar rho(cost_change / model_cost_change);
          if (rho > 0) {
            // Accept the Levenberg-Marquardt step because the linear
            // model fits well.
            x = x_new_;

            suc = Update(function, x);
            if (!suc) {
              summary.status = COST_FUNCTION_FAIL;
              break;
            }
            if (summary.gradient_max_norm < options.gradient_tolerance) {
              summary.status = GRADIENT_TOO_SMALL;
              break;
            }

            if (cost_ < options.cost_threshold) {
              summary.status = COST_TOO_SMALL;
              break;
            }

            Scalar tmp = Scalar(2 * rho - 1);
            u = u * std::max(1 / 3., 1 - tmp * tmp * tmp);
            v = 2;
            continue;
          }

          // Reject the update because either the normal equations failed to
          // solve or the local linear model was not good (rho < 0). Instead,
          // increase u to move closer to gradient descent.
          u *= v;
          v *= 2;
        }

        break;//end of LM
      }

    } //end of switch

    summary.final_cost = cost_;
    return summary;
  }

  Options options;
  Summary summary;

 private:
  // Preallocate everything, including temporary storage needed for solving the
  // linear system. This allows reusing the intermediate storage across solves.
  LinearSolver linear_solver_;
  Scalar cost_;
  Parameters dx_, x_new_, g_, jacobi_scaling_, lm_diagonal_, lm_step_;
  Eigen::Matrix<Scalar, NUM_RESIDUALS, 1> error_, f_x_new_, weights_;
  Eigen::Matrix<Scalar, NUM_RESIDUALS, NUM_PARAMETERS> jacobian_;
  Eigen::Matrix<Scalar, NUM_PARAMETERS, NUM_PARAMETERS> jtj_, jtj_regularized_;

  // The following definitions are needed for template metaprogramming.
  template<bool Condition, typename T>
  struct enable_if;

  template<typename T>
  struct enable_if<true, T> {
    typedef T type;
  };



  //Template magic to deal with two types of cost functions
  // 1) is the standard ceres cost function computing residuals and
  // and the jacobians.
  // 2) You compute the hessian and gradient directly.

  template<typename TFunc>
  typename std::enable_if<std::is_same<typename std::result_of<
      TFunc(const double *parameters,
            double *residuals,
            double *jacobian)>::type, bool>::value, bool>::type
  UpdateCostFunction(TFunc &func, const Parameters &x, bool only_cost) {

    if (only_cost) {
      return func(x.data(), f_x_new_.data(), NULL);
    }

    if (!func(x.data(), error_.data(), jacobian_.data())) {
      return false;
    }

    error_ = -error_;

    // On the first iteration, compute a diagonal (Jacobi) scaling
    // matrix, which we store as a vector.
    if (summary.iterations == 0) {
      // jacobi_scaling = 1 / (1 + diagonal(J'J))
      //
      // 1 is added to the denominator to regularize small diagonal
      // entries.
      jacobi_scaling_ = 1.0 / (1.0 + jacobian_.colwise().norm().array());
      jacobi_scaling_.setConstant(1.0);
    }

    // This explicitly computes the normal equations, which is numerically
    // unstable. Nevertheless, it is often good enough and is fast.
    //
    // TODO(sameeragarwal): Refactor this to allow for DenseQR
    // factorization.
    //jacobian_ = jacobian_ * jacobi_scaling_.asDiagonal();
    jtj_ = jacobian_.transpose() * jacobian_;
    g_ = jacobian_.transpose() * error_;
    summary.gradient_max_norm = g_.array().abs().maxCoeff();
    cost_ = error_.squaredNorm() / 2;
    return true;

  }

  template<typename TFunc>
  typename std::enable_if<std::is_same<typename std::result_of<
      TFunc(const double *parameters,
            double *residuals,
            double *gradient,
            double *hessian)>::type, bool>::value, bool>::type
  UpdateCostFunction(TFunc &func, const Parameters &x, bool only_cost) {
    //Only compute the cost(error) with given parameter
    if (only_cost) {
      return func(x.data(), f_x_new_.data(), NULL, NULL);
    }

    //this keeps the end user safe from accidentally not resetting the matrices
    jtj_.setZero();
    g_.setZero();
    //Call the cost function that automatically fills the hessian JtJ and the
    // gradient g_
    if (!func(x.data(), error_.data(), g_.data(), jtj_.data())) {
      return false;
    }

    if (summary.iterations == 0) {
      //no scaling. End user is responsible for making sure the hessian
      //is properly conditioned
      jacobi_scaling_.setConstant(1.0);
    }


    error_ = -error_;
    summary.gradient_max_norm = g_.array().abs().maxCoeff();
    cost_ = error_.squaredNorm() / 2;
    return true;

  }

  // The number of parameters and residuals are dynamically sized.
  template<int R, int P>
  typename enable_if<(R == Eigen::Dynamic && P == Eigen::Dynamic), void>::type
  Initialize(const CostFunction &function) {
    Initialize(function.NumResiduals(), function.NumParameters());
  }

  // The number of parameters is dynamically sized and the number of
  // residuals is statically sized.
  template<int R, int P>
  typename enable_if<(R == Eigen::Dynamic && P != Eigen::Dynamic), void>::type
  Initialize(const CostFunction &function) {
    Initialize(function.NumResiduals(), P);
  }

  // The number of parameters is statically sized and the number of
  // residuals is dynamically sized.
  template<int R, int P>
  typename enable_if<(R != Eigen::Dynamic && P == Eigen::Dynamic), void>::type
  Initialize(const CostFunction &function) {
    Initialize(R, function.NumParameters());
  }

  // The number of parameters and residuals are statically sized.
  template<int R, int P>
  typename enable_if<(R != Eigen::Dynamic && P != Eigen::Dynamic), void>::type
  Initialize(const CostFunction & /* function */) {}

  void Initialize(int num_residuals, int num_parameters) {
    dx_.resize(num_parameters);
    x_new_.resize(num_parameters);
    g_.resize(num_parameters);
    jacobi_scaling_.resize(num_parameters);
    lm_diagonal_.resize(num_parameters);
    lm_step_.resize(num_parameters);
    error_.resize(num_residuals);
    f_x_new_.resize(num_residuals);
    weights_.resize(num_residuals);
    jacobian_.resize(num_residuals, num_parameters);
    jtj_.resize(num_parameters, num_parameters);
    jtj_regularized_.resize(num_parameters, num_parameters);
  }
};

}  // namespace ts

