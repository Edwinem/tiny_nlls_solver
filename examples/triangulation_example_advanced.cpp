

/**
 *
 * This is the same as triangulation_example.cpp except it builds the hessian
 * matrix directly which allows for some more advanced tricks such as the huber
 * norm.
 *
 * The formulation of this version comes from the triangulation method first
 * introduced in the MSCKF paper:
 *
 * A. I. Mourikis and S. I. Roumeliotis, "A Multi-State Constraint Kalman Filter
 * for Vision-aided Inertial Navigation," Proceedings 2007 IEEE International
 * Conference on Robotics and Automation, Roma, 2007, pp. 3565-3572.
 * doi: 10.1109/ROBOT.2007.364024
 *
 */


#include <Eigen/Geometry>
#include <iostream>
#include <tiny_solver.h>
using Vec2d=Eigen::Vector2d;
using Vec3d=Eigen::Vector3d;
using VecXd=Eigen::VectorXd;
using MatXd=Eigen::MatrixXd;

/**
 * Compute the Reprojection error between a 3D point and its projection
 * into a camera position
 * \param pose Camera position
 * \param pt_msckf Point in MSCKF format alpha,beta,rho
 * \param proj Normalized Coordinates image projection
 * \return ReprojectionError
 */
Vec2d ReprojectionError(const Eigen::Isometry3d &pose,
                        const Vec3d &pt_msckf,
                        const Vec2d &proj) {
  const double alpha = pt_msckf(0);
  const double beta = pt_msckf(1);
  const double rho = pt_msckf(2);

  Vec3d h = pose.linear() * Vec3d(alpha, beta, 1.0) + rho * pose.translation();

  Vec2d z_hat(h[0] / h[2], h[1] / h[2]);
  return (z_hat - proj);
}

/**
 * Compute the Jacobian of a 3D point
 * \param T_c0_ci Position of the camera
 * \param x Point MSCKF format
 * \param z Measurement/Projection in normalized Image coordinates
 * \param J 2x3 Jacobian is returned here
 */
void FeatureOptJacobian(const Eigen::Isometry3d &T_c0_ci,
                        const Vec3d &x,
                        const Vec2d &z,
                        Eigen::Matrix<double, 2, 3> &J,
                        Eigen::Vector2d &res,
                        double &w) {

  // Compute hi1, hi2, and hi3 as Equation (37).
  const double &alpha = x(0);
  const double &beta = x(1);
  const double &rho = x(2);

  Vec3d h =
      T_c0_ci.linear() * Vec3d(alpha, beta, 1.0) + rho * T_c0_ci.translation();
  double &h1 = h(0);
  double &h2 = h(1);
  double &h3 = h(2);

  // Compute the Jacobian.
  Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
  W.leftCols<2>() = T_c0_ci.linear().leftCols<2>();
  W.rightCols<1>() = T_c0_ci.translation();

  J = MatXd::Zero(2, 3);
  J.row(0) = 1 / h3 * W.row(0) - h1 / (h3 * h3) * W.row(2);
  J.row(1) = 1 / h3 * W.row(1) - h2 / (h3 * h3) * W.row(2);

  // Compute the residual.
  Vec2d z_hat(h1 / h3, h2 / h3);
  res = z_hat - z;


//   Compute the weight based on the residual.
  double e = res.norm();
  double huber_epsilon = 0.01;
  if (e <= huber_epsilon) {
    w = 1.0;
  } else {
    w = huber_epsilon / (2 * e);
  }

}

struct TriangulationCostFunction {
  TriangulationCostFunction(
      std::vector<Eigen::Isometry3d,
                  Eigen::aligned_allocator<Eigen::Isometry3d>> &poses,
      std::vector<Eigen::Vector2d,
                  Eigen::aligned_allocator<Eigen::Vector2d>> &projections) {
    this->projections = projections;
    this->poses = poses;

  }

  typedef double Scalar;
  enum {
    NUM_RESIDUALS = Eigen::Dynamic,
    NUM_PARAMETERS = 3,
  };

  int NumResiduals() const {
    return poses.size() * 2; //Each camera pose has 2 residuals: x and y
  }

  bool operator()(const double *parameters,
                  double *residuals,
                  double *gradient,
                  double *hessian) const {

    Eigen::Map<const Eigen::Vector3d> point(parameters);

    for (int idx = 0; idx < poses.size(); ++idx) {
      Eigen::Vector2d
          error = ReprojectionError(poses[idx], point, projections[idx]);
      residuals[idx * 2 + 0] = error[0];
      residuals[idx * 2 + 1] = error[1];
    }

    if (gradient && hessian) {

      Eigen::Map<Eigen::Matrix<Scalar, NUM_PARAMETERS, 1>>
          b(gradient, NUM_PARAMETERS, 1);
      Eigen::Map<Eigen::Matrix<Scalar, NUM_PARAMETERS, NUM_PARAMETERS>>
          A(hessian, NUM_PARAMETERS, NUM_PARAMETERS);
      A.setZero();
      b.setZero();

      Eigen::Matrix<double, 2, 3> J_work = MatXd::Zero(2, 3);
      for (int idx = 0; idx < poses.size(); ++idx) {
        double weight;
        Eigen::Vector2d res;
        FeatureOptJacobian(poses[idx], point, projections[idx], J_work, res,
                           weight);

        //Utilize the weighting from the calculated huber norm
        if (weight == 1) {
          A += J_work.transpose() * J_work;
          b += J_work.transpose() * -res;
        } else {
          double w_square = weight * weight;
          A += w_square * J_work.transpose() * J_work;
          b += w_square * J_work.transpose() * -res;
        }
      }
    }

    return true;
  }

  std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>
      poses;
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      projections;

};

Eigen::Vector2d reprojectPoint(
    const Eigen::Isometry3d &G_T_C, const Eigen::Vector3d &G_p_fi) {
  const Eigen::Vector3d C_p_fi = G_T_C.inverse() * G_p_fi;
  return C_p_fi.hnormalized().head<2>();
}

int main() {

  const int number_view = 5;
  Eigen::Quaterniond rotations[] = {
      Eigen::Quaterniond(1, 0, 0, 0),
      Eigen::Quaterniond(
          Eigen::Quaterniond(
              Eigen::AngleAxisd(0.15, Eigen::Vector3d(0.0, 1.0, 0.1)))
              .normalized()),
      Eigen::Quaterniond(
          Eigen::Quaterniond(
              Eigen::AngleAxisd(0.05, Eigen::Vector3d(0.3, 1.0, 0.0)))
              .normalized()),
      Eigen::Quaterniond(
          Eigen::Quaterniond(
              Eigen::AngleAxisd(0.15, Eigen::Vector3d(0.2, 0.3, 0.1)))
              .normalized()),
      Eigen::Quaterniond(
          Eigen::Quaterniond(
              Eigen::AngleAxisd(-0.1, Eigen::Vector3d(0.1, 1.0, 0.0)))
              .normalized())};

  Vec3d positions[] = {
      Vec3d(0, 0, 0), Vec3d(-3, 0, 0),
      Vec3d(0.85, 0.1, -0.3), Vec3d(-0.1, -0.05, 0.4),
      Vec3d(0.7, 0.3, 0.21)};

  const Vec3d pt3d(1.5, 0.0, 4.0);

  std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>
      poses;

  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      projections;
  for (int idx = 0; idx < 5; ++idx) {
    Eigen::Matrix4d pose;
    pose.block<3, 3>(0, 0) = rotations[idx].toRotationMatrix();
    pose.block<3, 1>(0, 3) = positions[idx];
    Eigen::Isometry3d iso(pose);
    poses.push_back(iso);
    projections.push_back(reprojectPoint(iso, pt3d));
  }

  //Convert the poses so that they are relative to the first pose so that
  // pose[0] is the identity matrix
  Eigen::Isometry3d T_c0_w = poses[0];
  for (auto &pose : poses) {
    pose = pose.inverse() * T_c0_w;

  }

  ts::TinySolver<TriangulationCostFunction> solver;

  TriangulationCostFunction cost_functor(poses, projections);

  Vec3d initial_guess(1.4, 0.1, 4.2);
  Vec3d copy=initial_guess;

  //Convert to msckf form alpha,beta,rho
  initial_guess /= initial_guess[2];
  initial_guess[2] = 1.0 / initial_guess[2];
  auto summary=solver.Solve(cost_functor, &initial_guess);

  //Convert MSCKF form alpha,beta,rho back to normal 3D
  Eigen::Vector3d final_position(initial_guess(0) / initial_guess(2),
                                 initial_guess(1) / initial_guess(2),
                                 1.0 / initial_guess(2));

  //We triangulate it with the first position as the origin. Here we undo this
  final_position = poses[0].linear() * final_position + poses[0].translation();

  std::cout <<  "True answer is " << pt3d.transpose() << "\n" <<
            "Started with initial guess of " << copy.transpose() << "\n" <<
            "Optimized to a position of " << final_position.transpose() <<
            " with a final cost of "<< summary.final_cost << ".\n";

}