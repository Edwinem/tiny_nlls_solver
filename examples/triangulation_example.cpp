


/**
 *
 * This example contains a solver to triangulate a 3D point viewed from multiple
 * camera poses.
 *
 * The formulation of this version comes from the triangulation method first
 * introduced in the MSCKF paper:
 *
 * A. I. Mourikis and S. I. Roumeliotis, "A Multi-State Constraint Kalman Filter
 * for Vision-aided Inertial Navigation," Proceedings 2007 IEEE International
 * Conference on Robotics and Automation, Roma, 2007, pp. 3565-3572.
 * doi: 10.1109/ROBOT.2007.364024
 *
 *
 *
 *
 *
 *
 *
 */


#include <Eigen/Geometry>
using Vec2d=Eigen::Vector2d;
using Vec3d=Eigen::Vector3d;
using VecXd=Eigen::VectorXd;
using MatXd=Eigen::MatrixXd;

/**
 * Compute the Reprojection error between a 3D point and its projection
 * into a camera position
 * \param pose Camera position
 * \param pt3d 3D point
 * \param proj Normalized Coordinates image projection
 * \return ReprojectionError
 */
Vec2d ReprojectionError(const Eigen::Isometry3d &pose,
                        const Vec3d &pt3d,
                        const Vec2d &proj) {
  const double alpha = pt3d(0);
  const double beta = pt3d(0);
  const double rho = pt3d(0);

  Vec3d h = pose.linear() * Vec3d(alpha, beta, 1.0) + rho * pose.translation();

  Vec2d z_hat(h[0] / h[2], h[1] / h[2]);
  return (z_hat - proj);
}

/**
 * Compute the Jacobian of a 3D point
 * \param T_c0_ci Position of the camera
 * \param x 3D point
 * \param z Measurement/Projection in normalized Image coordinates
 * \param J 2x3 Jacobian is returned here
 */
void FeatureOptJacobian(const Eigen::Isometry3d &T_c0_ci,
                        const Vec3d &x,
                        const Vec2d &z,
                        MatXd &J) {

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


//  // Compute the weight based on the residual.
//  double e             = r.norm();
//  double huber_epsilon = 0.01;
//  if (e <= huber_epsilon) {
//    w = 1.0;
//  } else {
//    w = huber_epsilon / (2 * e);
//  }
}

struct TriangulationCostFunction {
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
                  double *jacobian) const {

    Eigen::Map<const Eigen::Vector3d> point(parameters);

    for (int idx = 0; idx < poses.size(); ++idx) {
      Eigen::Vector2d
          error = ReprojectionError(poses[idx], point, projections[idx]);
      residuals[idx * 2 + 0] = error[0];
      residuals[idx * 2 + 1] = error[1];
    }

    if (jacobian) {

      Eigen::Map<Eigen::Matrix<Scalar, NUM_RESIDUALS, NUM_PARAMETERS>>
          jac (jacobian, NumResiduals(), NUM_PARAMETERS);

      MatXd J_work = MatXd::Zero(2, 3);
      for (int idx = 0; idx < poses.size(); ++idx) {
        FeatureOptJacobian(poses[0], point, projections[idx], J_work);
        jac.block(idx*2,0,2,3)=J_work;
      }
    }

    return true;
  }

 private:

  std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>
      poses;
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      projections;

};

int main() {

}