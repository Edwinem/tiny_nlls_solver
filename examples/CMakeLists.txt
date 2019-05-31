


option(WITH_OPEN3D    "Turn on the examples"                             OFF)

if(${WITH_OPEN3D})
find_package(Open3D)

include_directories(${Open3D_INCLUDE_DIRS})
link_directories(${Open3D_LIBRARY_DIRS})
add_executable(dense_visual_odom_open3d dense_visual_odometry_open3d.cpp)

target_link_libraries(dense_visual_odom_open3d ${Open3D_LIBRARIES} -gomp -fopenmp)
endif()


add_executable(triangulation triangulation_example.cpp)