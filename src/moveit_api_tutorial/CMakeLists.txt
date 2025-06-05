cmake_minimum_required(VERSION 3.8)
project(moveit_api_tutorial)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(moveit_common REQUIRED)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_visual_tools REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rviz_visual_tools REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)


set(THIS_PACKAGE_INCLUDE_DEPENDS
  ament_cmake
  moveit_core
  moveit_msgs
  moveit_ros_planning
  moveit_ros_planning_interface
  moveit_visual_tools
  rclcpp
  rviz_visual_tools
  tf2_geometry_msgs
  tf2_ros  
)

add_executable(moveit_cpp_interface src/moveit_cpp_interface.cpp)
ament_target_dependencies(moveit_cpp_interface  ${THIS_PACKAGE_INCLUDE_DEPENDS})

install(TARGETS 
  moveit_cpp_interface
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
