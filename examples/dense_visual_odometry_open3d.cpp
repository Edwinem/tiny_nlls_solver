//
// Created by nikolausmitchell on 5/29/19.
//


#include "Open3D/Open3D.h"
#include <memory>
#include <tiny_solver.h>

using namespace open3d;
using namespace odometry;
using namespace geometry;

std::shared_ptr<geometry::Image> PreprocessDepth(
    const geometry::Image &depth_orig, const OdometryOption &option);
std::shared_ptr<CorrespondenceSetPixelWise> ComputeCorrespondence(
    const Eigen::Matrix3d intrinsic_matrix,
    const Eigen::Matrix4d &extrinsic,
    const geometry::Image &depth_s,
    const geometry::Image &depth_t,
    const OdometryOption &option);
inline std::shared_ptr<geometry::RGBDImage> PackRGBDImage(
    const geometry::Image &color, const geometry::Image &depth);

void NormalizeIntensity(geometry::Image &image_s,
                        geometry::Image &image_t,
                        CorrespondenceSetPixelWise &correspondence);
std::vector<Eigen::Matrix3d> CreateCameraMatrixPyramid(
    const camera::PinholeCameraIntrinsic &pinhole_camera_intrinsic,
    int levels);
std::shared_ptr<geometry::Image> ConvertDepthImageToXYZImage(
    const geometry::Image &depth, const Eigen::Matrix3d &intrinsic_matrix);

struct DenseCostFunction {

  typedef double Scalar;
  enum {
    NUM_RESIDUALS = Eigen::Dynamic,
    NUM_PARAMETERS = 6,
  };

  int NumResiduals() const {
    return source_pyramid[cur_level]->color_.width_
        * source_pyramid[0]->color_.height_;
  }

  DenseCostFunction(const geometry::RGBDImage &source,
                    const geometry::RGBDImage &target,
                    const camera::PinholeCameraIntrinsic &pinhole_camera_intrinsic,
                    const Eigen::Matrix4d &odo_init,
                    const OdometryOption &option) {
    this->options = option;

    jacobian_method =
        std::unique_ptr<RGBDOdometryJacobian>(new RGBDOdometryJacobianFromColorTerm());

    auto source_gray = geometry::FilterImage(
        source.color_, geometry::Image::FilterType::Gaussian3);
    auto target_gray = geometry::FilterImage(
        target.color_, geometry::Image::FilterType::Gaussian3);
    auto source_depth_preprocessed = PreprocessDepth(source.depth_, option);
    auto target_depth_preprocessed = PreprocessDepth(target.depth_, option);
    auto source_depth = geometry::FilterImage(
        *source_depth_preprocessed, geometry::Image::FilterType::Gaussian3);
    auto target_depth = geometry::FilterImage(
        *target_depth_preprocessed, geometry::Image::FilterType::Gaussian3);

    auto correspondence = ComputeCorrespondence(
        pinhole_camera_intrinsic.intrinsic_matrix_, odo_init, *source_depth,
        *target_depth, option);
    NormalizeIntensity(*source_gray, *target_gray, *correspondence);

    auto processed_source = PackRGBDImage(*source_gray, *source_depth);
    auto processed_target = PackRGBDImage(*target_gray, *target_depth);

    std::vector<int> iter_counts = option.iteration_number_per_pyramid_level_;
    int num_levels = (int) iter_counts.size();

    source_pyramid =
        geometry::CreateRGBDImagePyramid(*processed_source, num_levels);
    target_pyramid =
        geometry::CreateRGBDImagePyramid(*processed_target, num_levels);
    target_pyramid_dx = geometry::FilterRGBDImagePyramid(
        target_pyramid, geometry::Image::FilterType::Sobel3Dx);
    target_pyramid_dy = geometry::FilterRGBDImagePyramid(
        target_pyramid, geometry::Image::FilterType::Sobel3Dy);

    pyramid_camera_matrix = CreateCameraMatrixPyramid(pinhole_camera_intrinsic,
                                                      (int) iter_counts.size());

  }

  void CreateCurrentLevelImages(int cur_level) {
    this->cur_level = cur_level;
    cur_level_camera_matrix =
        pyramid_camera_matrix[cur_level];
    source_xyz_cur_level = ConvertDepthImageToXYZImage(
        source_pyramid[cur_level]->depth_, cur_level_camera_matrix);
    source_cur_level = PackRGBDImage(source_pyramid[cur_level]->color_,
                                     source_pyramid[cur_level]->depth_);
    target_cur_level = PackRGBDImage(target_pyramid[cur_level]->color_,
                                     target_pyramid[cur_level]->depth_);
    target_dx_cur_level = PackRGBDImage(target_pyramid_dx[cur_level]->color_,
                                        target_pyramid_dx[cur_level]->depth_);
    target_dy_cur_level = PackRGBDImage(target_pyramid_dy[cur_level]->color_,
                                        target_pyramid_dy[cur_level]->depth_);
  }

  bool operator()(const double *parameters,
                  double *residuals,
                  double *jacobian) {

    const Eigen::Map<const Eigen::Vector6d> params(parameters);
    Eigen::Matrix4d curr_odo = utility::TransformVector6dToMatrix4d(params);
    auto correspondence = ComputeCorrespondence(
        cur_level_camera_matrix,
        curr_odo,
        source_cur_level->depth_,
        target_cur_level->depth_,
        options);
    int corresps_count = (int) correspondence->size();

    auto f_lambda =
        [&](int i,
            std::vector<Eigen::Vector6d, utility::Vector6d_allocator> &J_r,
            std::vector<double> &r) {
          jacobian_method->ComputeJacobianAndResidual(
              i, J_r, r, *source_cur_level, *target_cur_level,
              *source_xyz_cur_level, *target_dx_cur_level,
              *target_dy_cur_level, cur_level_camera_matrix, curr_odo,
              *correspondence);
        };

    std::vector<Eigen::Vector6d, utility::Vector6d_allocator> J_r;
    std::vector<double> res;
    res.resize(NumResiduals());
    J_r.resize(NumResiduals());
    int i = 0;
    for (i = 0; i < corresps_count; i++) {

      f_lambda(i, J_r, res);
    }

    for (int j = i; i < NumResiduals(); ++j) {
      res[i] = 0;
      J_r[i] = Eigen::Vector6d::Zero();
    }

    for (int idx = 0; idx < NumResiduals(); ++idx) {
      residuals[idx] = res[idx];
    }

    if (jacobian) {
      Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
          jac(jacobian, NumResiduals(), NUM_PARAMETERS);
      for (int idx = 0; idx < NumResiduals(); ++idx) {
        jac.block(idx, 0, 1, 6) = J_r[idx];
      }
    }

  }

 private:

  OdometryOption options;
  std::unique_ptr<RGBDOdometryJacobian> jacobian_method;

  std::vector<std::shared_ptr<RGBDImage>> source_pyramid;
  std::vector<std::shared_ptr<RGBDImage>> target_pyramid;
  std::vector<std::shared_ptr<RGBDImage>> target_pyramid_dx;
  std::vector<std::shared_ptr<RGBDImage>> target_pyramid_dy;
  std::vector<Eigen::Matrix3d> pyramid_camera_matrix;

  int cur_level = 0;
  Eigen::Matrix3d cur_level_camera_matrix;
  std::shared_ptr<geometry::Image> source_xyz_cur_level;
  std::shared_ptr<geometry::RGBDImage> source_cur_level;
  std::shared_ptr<geometry::RGBDImage> target_cur_level;
  std::shared_ptr<geometry::RGBDImage> target_dx_cur_level;
  std::shared_ptr<geometry::RGBDImage> target_dy_cur_level;

};

struct CompositionParameterization {

  void operator()(
      const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 1>> x_prev,
      const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 1>> dx,
      Eigen::Ref<Eigen::Matrix<double, Eigen::Dynamic, 1>> x_new) {
    Eigen::Matrix4d prev = utility::TransformVector6dToMatrix4d(x_prev);
    Eigen::Matrix4d update = utility::TransformVector6dToMatrix4d(dx);
    Eigen::Matrix4d new_pose = prev * update;
    x_new = utility::TransformMatrix4dToVector6d(new_pose);
  }
};



void PrintHelp(char *argv[]) {
  using namespace open3d;

  PrintOpen3DVersion();
  // clang-format off
  utility::PrintInfo("Usage:\n");
  utility::PrintInfo(
      ">    OdometryRGBD [color_source] [source_target] [color_target] [depth_target] [options]\n");
  utility::PrintInfo("     Given RGBD image pair, estimate 6D odometry.\n");
  utility::PrintInfo("     [options]\n");
  utility::PrintInfo("     --camera_intrinsic [intrinsic_path]\n");
  utility::PrintInfo(
      "     --rgbd_type [number] (0:Redwood, 1:TUM, 2:SUN, 3:NYU)\n");
  utility::PrintInfo(
      "     --verbose : indicate this to display detailed information\n");
  utility::PrintInfo("     --hybrid : compute odometry using hybrid objective\n");
  // clang-format on
  utility::PrintInfo("\n");
}

int main(int argc, char *argv[]) {
  using namespace open3d;

  if (argc <= 4 || utility::ProgramOptionExists(argc, argv, "--help") ||
      utility::ProgramOptionExists(argc, argv, "-h")) {
    PrintHelp(argv);
    return 1;
  }

  std::string intrinsic_path;
  if (utility::ProgramOptionExists(argc, argv, "--camera_intrinsic")) {
    intrinsic_path = utility::GetProgramOptionAsString(argc, argv,
                                                       "--camera_intrinsic")
        .c_str();
    utility::PrintInfo("Camera intrinsic path %s\n",
                       intrinsic_path.c_str());
  } else {
    utility::PrintInfo("Camera intrinsic path is not given\n");
  }
  camera::PinholeCameraIntrinsic intrinsic;
  if (intrinsic_path.empty() ||
      !io::ReadIJsonConvertible(intrinsic_path, intrinsic)) {
    utility::PrintWarning(
        "Failed to read intrinsic parameters for depth image.\n");
    utility::PrintWarning("Use default value for Primesense camera.\n");
    intrinsic = camera::PinholeCameraIntrinsic(
        camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);
  }

  if (utility::ProgramOptionExists(argc, argv, "--verbose"))
    utility::SetVerbosityLevel(utility::VerbosityLevel::VerboseAlways);

  int rgbd_type =
      utility::GetProgramOptionAsInt(argc, argv, "--rgbd_type", 0);
  auto color_source = io::CreateImageFromFile(argv[1]);
  auto depth_source = io::CreateImageFromFile(argv[2]);
  auto color_target = io::CreateImageFromFile(argv[3]);
  auto depth_target = io::CreateImageFromFile(argv[4]);
  std::shared_ptr<geometry::RGBDImage> (*CreateRGBDImage)(
      const geometry::Image &, const geometry::Image &, bool);
  if (rgbd_type == 0)
    CreateRGBDImage = &geometry::CreateRGBDImageFromRedwoodFormat;
  else if (rgbd_type == 1)
    CreateRGBDImage = &geometry::CreateRGBDImageFromTUMFormat;
  else if (rgbd_type == 2)
    CreateRGBDImage = &geometry::CreateRGBDImageFromSUNFormat;
  else if (rgbd_type == 3)
    CreateRGBDImage = &geometry::CreateRGBDImageFromNYUFormat;
  else
    CreateRGBDImage = &geometry::CreateRGBDImageFromRedwoodFormat;
  auto source = CreateRGBDImage(*color_source, *depth_source, true);
  auto target = CreateRGBDImage(*color_target, *depth_target, true);

  odometry::OdometryOption option;
  Eigen::Matrix4d odo_init = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d trans_odo = Eigen::Matrix4d::Identity();
  Eigen::Matrix6d info_odo = Eigen::Matrix6d::Zero();

  DenseCostFunction cost_func(*source, *target, intrinsic, odo_init, option);

  ts::TinySolver<DenseCostFunction,Eigen::LDLT<
      Eigen::Matrix<typename DenseCostFunction::Scalar,
                    DenseCostFunction::NUM_PARAMETERS,
                    DenseCostFunction::NUM_PARAMETERS> >,
                    CompositionParameterization> tiny_solver;

  //For dense we do course to fine

  int n_levels = option.iteration_number_per_pyramid_level_.size();

  Eigen::Vector6d initial_estimate;

  for (int level = n_levels - 1; level >= 0; --level) {
    cost_func.CreateCurrentLevelImages(level);

    tiny_solver.Solve(cost_func, &initial_estimate);

  }

}

std::tuple<std::shared_ptr<geometry::Image>, std::shared_ptr<geometry::Image>>
InitializeCorrespondenceMap(int width, int height) {
  // initialization: filling with any (u,v) to (-1,-1)
  auto correspondence_map = std::make_shared<geometry::Image>();
  auto depth_buffer = std::make_shared<geometry::Image>();
  correspondence_map->PrepareImage(width, height, 2, 4);
  depth_buffer->PrepareImage(width, height, 1, 4);
  for (int v = 0; v < correspondence_map->height_; v++) {
    for (int u = 0; u < correspondence_map->width_; u++) {
      *geometry::PointerAt<int>(*correspondence_map, u, v, 0) = -1;
      *geometry::PointerAt<int>(*correspondence_map, u, v, 1) = -1;
      *geometry::PointerAt<float>(*depth_buffer, u, v, 0) = -1.0f;
    }
  }
  return std::make_tuple(correspondence_map, depth_buffer);
}

inline void AddElementToCorrespondenceMap(geometry::Image &correspondence_map,
                                          geometry::Image &depth_buffer,
                                          int u_s,
                                          int v_s,
                                          int u_t,
                                          int v_t,
                                          float transformed_d_t) {
  int exist_u_t, exist_v_t;
  double exist_d_t;
  exist_u_t = *geometry::PointerAt<int>(correspondence_map, u_s, v_s, 0);
  exist_v_t = *geometry::PointerAt<int>(correspondence_map, u_s, v_s, 1);
  if (exist_u_t != -1 && exist_v_t != -1) {
    exist_d_t = *geometry::PointerAt<float>(depth_buffer, u_s, v_s);
    if (transformed_d_t <
        exist_d_t) {  // update nearer point as correspondence
      *geometry::PointerAt<int>(correspondence_map, u_s, v_s, 0) = u_t;
      *geometry::PointerAt<int>(correspondence_map, u_s, v_s, 1) = v_t;
      *geometry::PointerAt<float>(depth_buffer, u_s, v_s) =
          transformed_d_t;
    }
  } else {  // register correspondence
    *geometry::PointerAt<int>(correspondence_map, u_s, v_s, 0) = u_t;
    *geometry::PointerAt<int>(correspondence_map, u_s, v_s, 1) = v_t;
    *geometry::PointerAt<float>(depth_buffer, u_s, v_s) = transformed_d_t;
  }
}

void MergeCorrespondenceMaps(geometry::Image &correspondence_map,
                             geometry::Image &depth_buffer,
                             geometry::Image &correspondence_map_part,
                             geometry::Image &depth_buffer_part) {
  for (int v_s = 0; v_s < correspondence_map.height_; v_s++) {
    for (int u_s = 0; u_s < correspondence_map.width_; u_s++) {
      int u_t = *geometry::PointerAt<int>(correspondence_map_part, u_s,
                                          v_s, 0);
      int v_t = *geometry::PointerAt<int>(correspondence_map_part, u_s,
                                          v_s, 1);
      if (u_t != -1 && v_t != -1) {
        float transformed_d_t = *geometry::PointerAt<float>(
            depth_buffer_part, u_s, v_s);
        AddElementToCorrespondenceMap(correspondence_map, depth_buffer,
                                      u_s, v_s, u_t, v_t,
                                      transformed_d_t);
      }
    }
  }
}

int CountCorrespondence(const geometry::Image &correspondence_map) {
  int correspondence_count = 0;
  for (int v_s = 0; v_s < correspondence_map.height_; v_s++) {
    for (int u_s = 0; u_s < correspondence_map.width_; u_s++) {
      int u_t =
          *geometry::PointerAt<int>(correspondence_map, u_s, v_s, 0);
      int v_t =
          *geometry::PointerAt<int>(correspondence_map, u_s, v_s, 1);
      if (u_t != -1 && v_t != -1) {
        correspondence_count++;
      }
    }
  }
  return correspondence_count;
}

std::shared_ptr<CorrespondenceSetPixelWise> ComputeCorrespondence(
    const Eigen::Matrix3d intrinsic_matrix,
    const Eigen::Matrix4d &extrinsic,
    const geometry::Image &depth_s,
    const geometry::Image &depth_t,
    const OdometryOption &option) {
  const Eigen::Matrix3d K = intrinsic_matrix;
  const Eigen::Matrix3d K_inv = K.inverse();
  const Eigen::Matrix3d R = extrinsic.block<3, 3>(0, 0);
  const Eigen::Matrix3d KRK_inv = K * R * K_inv;
  Eigen::Vector3d Kt = K * extrinsic.block<3, 1>(0, 3);

  std::shared_ptr<geometry::Image> correspondence_map;
  std::shared_ptr<geometry::Image> depth_buffer;
  std::tie(correspondence_map, depth_buffer) =
      InitializeCorrespondenceMap(depth_t.width_, depth_t.height_);

#ifdef _OPENMP
#pragma omp parallel
  {
#endif
  std::shared_ptr<geometry::Image> correspondence_map_private;
  std::shared_ptr<geometry::Image> depth_buffer_private;
  std::tie(correspondence_map_private, depth_buffer_private) =
      InitializeCorrespondenceMap(depth_t.width_, depth_t.height_);
#ifdef _OPENMP
#pragma omp for nowait
#endif
  for (int v_s = 0; v_s < depth_s.height_; v_s++) {
    for (int u_s = 0; u_s < depth_s.width_; u_s++) {
      double d_s = *geometry::PointerAt<float>(depth_s, u_s, v_s);
      if (!std::isnan(d_s)) {
        Eigen::Vector3d uv_in_s =
            d_s * KRK_inv * Eigen::Vector3d(u_s, v_s, 1.0) + Kt;
        double transformed_d_s = uv_in_s(2);
        int u_t = (int) (uv_in_s(0) / transformed_d_s + 0.5);
        int v_t = (int) (uv_in_s(1) / transformed_d_s + 0.5);
        if (u_t >= 0 && u_t < depth_t.width_ && v_t >= 0 &&
            v_t < depth_t.height_) {
          double d_t =
              *geometry::PointerAt<float>(depth_t, u_t, v_t);
          if (!std::isnan(d_t) &&
              std::abs(transformed_d_s - d_t) <=
                  option.max_depth_diff_) {
            AddElementToCorrespondenceMap(
                *correspondence_map_private,
                *depth_buffer_private, u_s, v_s, u_t, v_t,
                (float) d_s);
          }
        }
      }
    }
  }
#ifdef _OPENMP
#pragma omp critical
  {
#endif
  MergeCorrespondenceMaps(*correspondence_map, *depth_buffer,
                          *correspondence_map_private,
                          *depth_buffer_private);
#ifdef _OPENMP
  }  //    omp critical
    }      //    omp parallel
#endif

  auto correspondence = std::make_shared<CorrespondenceSetPixelWise>();
  int correspondence_count = CountCorrespondence(*correspondence_map);
  correspondence->resize(correspondence_count);
  int cnt = 0;
  for (int v_s = 0; v_s < correspondence_map->height_; v_s++) {
    for (int u_s = 0; u_s < correspondence_map->width_; u_s++) {
      int u_t =
          *geometry::PointerAt<int>(*correspondence_map, u_s, v_s, 0);
      int v_t =
          *geometry::PointerAt<int>(*correspondence_map, u_s, v_s, 1);
      if (u_t != -1 && v_t != -1) {
        Eigen::Vector4i pixel_correspondence(u_s, v_s, u_t, v_t);
        (*correspondence)[cnt] = pixel_correspondence;
        cnt++;
      }
    }
  }
  return correspondence;
}

std::shared_ptr<geometry::Image> ConvertDepthImageToXYZImage(
    const geometry::Image &depth, const Eigen::Matrix3d &intrinsic_matrix) {
  auto image_xyz = std::make_shared<geometry::Image>();
  if (depth.num_of_channels_ != 1 || depth.bytes_per_channel_ != 4) {
    utility::PrintDebug(
        "[ConvertDepthImageToXYZImage] Unsupported image format.\n");
    return image_xyz;
  }
  const double inv_fx = 1.0 / intrinsic_matrix(0, 0);
  const double inv_fy = 1.0 / intrinsic_matrix(1, 1);
  const double ox = intrinsic_matrix(0, 2);
  const double oy = intrinsic_matrix(1, 2);
  image_xyz->PrepareImage(depth.width_, depth.height_, 3, 4);

  for (int y = 0; y < image_xyz->height_; y++) {
    for (int x = 0; x < image_xyz->width_; x++) {
      float *px = geometry::PointerAt<float>(*image_xyz, x, y, 0);
      float *py = geometry::PointerAt<float>(*image_xyz, x, y, 1);
      float *pz = geometry::PointerAt<float>(*image_xyz, x, y, 2);
      float z = *geometry::PointerAt<float>(depth, x, y);
      *px = (float) ((x - ox) * z * inv_fx);
      *py = (float) ((y - oy) * z * inv_fy);
      *pz = z;
    }
  }
  return image_xyz;
}

std::vector<Eigen::Matrix3d> CreateCameraMatrixPyramid(
    const camera::PinholeCameraIntrinsic &pinhole_camera_intrinsic,
    int levels) {
  std::vector<Eigen::Matrix3d> pyramid_camera_matrix;
  pyramid_camera_matrix.reserve(levels);
  for (int i = 0; i < levels; i++) {
    Eigen::Matrix3d level_camera_matrix;
    if (i == 0)
      level_camera_matrix = pinhole_camera_intrinsic.intrinsic_matrix_;
    else
      level_camera_matrix = 0.5 * pyramid_camera_matrix[i - 1];
    level_camera_matrix(2, 2) = 1.;
    pyramid_camera_matrix.push_back(level_camera_matrix);
  }
  return pyramid_camera_matrix;
}

Eigen::Matrix6d CreateInformationMatrix(
    const Eigen::Matrix4d &extrinsic,
    const camera::PinholeCameraIntrinsic &pinhole_camera_intrinsic,
    const geometry::Image &depth_s,
    const geometry::Image &depth_t,
    const OdometryOption &option) {
  auto correspondence =
      ComputeCorrespondence(pinhole_camera_intrinsic.intrinsic_matrix_,
                            extrinsic, depth_s, depth_t, option);

  auto xyz_t = ConvertDepthImageToXYZImage(
      depth_t, pinhole_camera_intrinsic.intrinsic_matrix_);

  // write q^*
  // see http://redwood-data.org/indoor/registration.html
  // note: I comes first and q_skew is scaled by factor 2.
  Eigen::Matrix6d GTG = Eigen::Matrix6d::Identity();
#ifdef _OPENMP
#pragma omp parallel
  {
#endif
  Eigen::Matrix6d GTG_private = Eigen::Matrix6d::Identity();
  Eigen::Vector6d G_r_private = Eigen::Vector6d::Zero();
#ifdef _OPENMP
#pragma omp for nowait
#endif
  for (auto row = 0; row < correspondence->size(); row++) {
    int u_t = (*correspondence)[row](2);
    int v_t = (*correspondence)[row](3);
    double x = *geometry::PointerAt<float>(*xyz_t, u_t, v_t, 0);
    double y = *geometry::PointerAt<float>(*xyz_t, u_t, v_t, 1);
    double z = *geometry::PointerAt<float>(*xyz_t, u_t, v_t, 2);
    G_r_private.setZero();
    G_r_private(1) = z;
    G_r_private(2) = -y;
    G_r_private(3) = 1.0;
    GTG_private.noalias() += G_r_private * G_r_private.transpose();
    G_r_private.setZero();
    G_r_private(0) = -z;
    G_r_private(2) = x;
    G_r_private(4) = 1.0;
    GTG_private.noalias() += G_r_private * G_r_private.transpose();
    G_r_private.setZero();
    G_r_private(0) = y;
    G_r_private(1) = -x;
    G_r_private(5) = 1.0;
    GTG_private.noalias() += G_r_private * G_r_private.transpose();
  }
#ifdef _OPENMP
#pragma omp critical
#endif
  { GTG += GTG_private; }
#ifdef _OPENMP
  }
#endif
  return std::move(GTG);
}

void NormalizeIntensity(geometry::Image &image_s,
                        geometry::Image &image_t,
                        CorrespondenceSetPixelWise &correspondence) {
  if (image_s.width_ != image_t.width_ ||
      image_s.height_ != image_t.height_) {
    utility::PrintError(
        "[NormalizeIntensity] Size of two input images should be "
        "same\n");
    return;
  }
  double mean_s = 0.0, mean_t = 0.0;
  for (int row = 0; row < correspondence.size(); row++) {
    int u_s = correspondence[row](0);
    int v_s = correspondence[row](1);
    int u_t = correspondence[row](2);
    int v_t = correspondence[row](3);
    mean_s += *geometry::PointerAt<float>(image_s, u_s, v_s);
    mean_t += *geometry::PointerAt<float>(image_t, u_t, v_t);
  }
  mean_s /= (double) correspondence.size();
  mean_t /= (double) correspondence.size();
  geometry::LinearTransformImage(image_s, 0.5 / mean_s, 0.0);
  geometry::LinearTransformImage(image_t, 0.5 / mean_t, 0.0);
}

inline std::shared_ptr<geometry::RGBDImage> PackRGBDImage(
    const geometry::Image &color, const geometry::Image &depth) {
  return std::make_shared<geometry::RGBDImage>(
      geometry::RGBDImage(color, depth));
}

std::shared_ptr<geometry::Image> PreprocessDepth(
    const geometry::Image &depth_orig, const OdometryOption &option) {
  std::shared_ptr<geometry::Image> depth_processed =
      std::make_shared<geometry::Image>();
  *depth_processed = depth_orig;
  for (int y = 0; y < depth_processed->height_; y++) {
    for (int x = 0; x < depth_processed->width_; x++) {
      float *p = geometry::PointerAt<float>(*depth_processed, x, y);
      if ((*p < option.min_depth_ || *p > option.max_depth_ || *p <= 0))
        *p = std::numeric_limits<float>::quiet_NaN();
    }
  }
  return depth_processed;
}

inline bool CheckImagePair(const geometry::Image &image_s,
                           const geometry::Image &image_t) {
  return (image_s.width_ == image_t.width_ &&
      image_s.height_ == image_t.height_);
}

inline bool CheckRGBDImagePair(const geometry::RGBDImage &source,
                               const geometry::RGBDImage &target) {
  return (CheckImagePair(source.color_, target.color_) &&
      CheckImagePair(source.depth_, target.depth_) &&
      CheckImagePair(source.color_, source.depth_) &&
      CheckImagePair(target.color_, target.depth_) &&
      CheckImagePair(source.color_, target.color_) &&
      source.color_.num_of_channels_ == 1 &&
      source.depth_.num_of_channels_ == 1 &&
      target.color_.num_of_channels_ == 1 &&
      target.depth_.num_of_channels_ == 1 &&
      source.color_.bytes_per_channel_ == 4 &&
      target.color_.bytes_per_channel_ == 4 &&
      source.depth_.bytes_per_channel_ == 4 &&
      target.depth_.bytes_per_channel_ == 4);
}

std::tuple<std::shared_ptr<geometry::RGBDImage>,
           std::shared_ptr<geometry::RGBDImage>>
InitializeRGBDOdometry(
    const geometry::RGBDImage &source,
    const geometry::RGBDImage &target,
    const camera::PinholeCameraIntrinsic &pinhole_camera_intrinsic,
    const Eigen::Matrix4d &odo_init,
    const OdometryOption &option) {
  auto source_gray = geometry::FilterImage(
      source.color_, geometry::Image::FilterType::Gaussian3);
  auto target_gray = geometry::FilterImage(
      target.color_, geometry::Image::FilterType::Gaussian3);
  auto source_depth_preprocessed = PreprocessDepth(source.depth_, option);
  auto target_depth_preprocessed = PreprocessDepth(target.depth_, option);
  auto source_depth = geometry::FilterImage(
      *source_depth_preprocessed, geometry::Image::FilterType::Gaussian3);
  auto target_depth = geometry::FilterImage(
      *target_depth_preprocessed, geometry::Image::FilterType::Gaussian3);

  auto correspondence = ComputeCorrespondence(
      pinhole_camera_intrinsic.intrinsic_matrix_, odo_init, *source_depth,
      *target_depth, option);
  NormalizeIntensity(*source_gray, *target_gray, *correspondence);

  auto source_out = PackRGBDImage(*source_gray, *source_depth);
  auto target_out = PackRGBDImage(*target_gray, *target_depth);
  return std::make_tuple(source_out, target_out);
}

std::tuple<bool, Eigen::Matrix4d> DoSingleIteration(
    int iter,
    int level,
    const geometry::RGBDImage &source,
    const geometry::RGBDImage &target,
    const geometry::Image &source_xyz,
    const geometry::RGBDImage &target_dx,
    const geometry::RGBDImage &target_dy,
    const Eigen::Matrix3d intrinsic,
    const Eigen::Matrix4d &extrinsic_initial,
    const RGBDOdometryJacobian &jacobian_method,
    const OdometryOption &option) {
  auto correspondence = ComputeCorrespondence(
      intrinsic, extrinsic_initial, source.depth_, target.depth_, option);
  int corresps_count = (int) correspondence->size();

  auto f_lambda =
      [&](int i,
          std::vector<Eigen::Vector6d, utility::Vector6d_allocator> &J_r,
          std::vector<double> &r) {
        jacobian_method.ComputeJacobianAndResidual(
            i, J_r, r, source, target, source_xyz, target_dx,
            target_dy, intrinsic, extrinsic_initial,
            *correspondence);
      };
  utility::PrintDebug("Iter : %d, Level : %d, ", iter, level);
  Eigen::Matrix6d JTJ;
  Eigen::Vector6d JTr;
  double r2;
  std::tie(JTJ, JTr, r2) =
      utility::ComputeJTJandJTr<Eigen::Matrix6d, Eigen::Vector6d>(
          f_lambda, corresps_count);

  bool is_success;
  Eigen::Matrix4d extrinsic;
  std::tie(is_success, extrinsic) =
      utility::SolveJacobianSystemAndObtainExtrinsicMatrix(JTJ, JTr);
  if (!is_success) {
    utility::PrintWarning("[ComputeOdometry] no solution!\n");
    return std::make_tuple(false, Eigen::Matrix4d::Identity());
  } else {
    return std::make_tuple(true, extrinsic);
  }
}

std::tuple<bool, Eigen::Matrix4d> ComputeMultiscale(
    const geometry::RGBDImage &source,
    const geometry::RGBDImage &target,
    const camera::PinholeCameraIntrinsic &pinhole_camera_intrinsic,
    const Eigen::Matrix4d &extrinsic_initial,
    const RGBDOdometryJacobian &jacobian_method,
    const OdometryOption &option) {
  std::vector<int> iter_counts = option.iteration_number_per_pyramid_level_;
  int num_levels = (int) iter_counts.size();

  auto source_pyramid = geometry::CreateRGBDImagePyramid(source, num_levels);
  auto target_pyramid = geometry::CreateRGBDImagePyramid(target, num_levels);
  auto target_pyramid_dx = geometry::FilterRGBDImagePyramid(
      target_pyramid, geometry::Image::FilterType::Sobel3Dx);
  auto target_pyramid_dy = geometry::FilterRGBDImagePyramid(
      target_pyramid, geometry::Image::FilterType::Sobel3Dy);

  Eigen::Matrix4d result_odo = extrinsic_initial.isZero()
                               ? Eigen::Matrix4d::Identity()
                               : extrinsic_initial;

  std::vector<Eigen::Matrix3d> pyramid_camera_matrix =
      CreateCameraMatrixPyramid(pinhole_camera_intrinsic,
                                (int) iter_counts.size());

  for (int level = num_levels - 1; level >= 0; level--) {
    const Eigen::Matrix3d level_camera_matrix =
        pyramid_camera_matrix[level];

    auto source_xyz_level = ConvertDepthImageToXYZImage(
        source_pyramid[level]->depth_, level_camera_matrix);
    auto source_level = PackRGBDImage(source_pyramid[level]->color_,
                                      source_pyramid[level]->depth_);
    auto target_level = PackRGBDImage(target_pyramid[level]->color_,
                                      target_pyramid[level]->depth_);
    auto target_dx_level = PackRGBDImage(target_pyramid_dx[level]->color_,
                                         target_pyramid_dx[level]->depth_);
    auto target_dy_level = PackRGBDImage(target_pyramid_dy[level]->color_,
                                         target_pyramid_dy[level]->depth_);

    for (int iter = 0; iter < iter_counts[num_levels - level - 1]; iter++) {
      Eigen::Matrix4d curr_odo;
      bool is_success;
      std::tie(is_success, curr_odo) = DoSingleIteration(
          iter, level, *source_level, *target_level,
          *source_xyz_level, *target_dx_level, *target_dy_level,
          level_camera_matrix, result_odo, jacobian_method, option);
      result_odo = curr_odo * result_odo;

      if (!is_success) {
        utility::PrintWarning("[ComputeOdometry] no solution!\n");
        return std::make_tuple(false, Eigen::Matrix4d::Identity());
      }
    }
  }
  return std::make_tuple(true, result_odo);
}