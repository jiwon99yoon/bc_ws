cmake_minimum_required(VERSION 3.8)
project(dm_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(Boost REQUIRED COMPONENTS python)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(eigenpy REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(pinocchio REQUIRED)

include_directories(
    include
    ${EIGEN3_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${eigenpy_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} SHARED
    src/robot_model/robot_updater.cpp    
    src/utils/dyros_math.cpp    
    src/controller.cpp
)

# Add the Python wrapper library
add_library(${PROJECT_NAME}_wrapper_cpp SHARED
    src/utils/bindings.cpp
)

target_link_libraries(${PROJECT_NAME} 
    Eigen3::Eigen
    rclcpp::rclcpp    
)

target_link_libraries(${PROJECT_NAME}_wrapper_cpp
    ${PROJECT_NAME}
    ${Boost_LIBRARIES}
    Eigen3::Eigen
    Python3::Python
    eigenpy::eigenpy
    pinocchio::pinocchio    
    rclcpp::rclcpp
)

# Set output properties for the Python module
set_target_properties(${PROJECT_NAME}_wrapper_cpp PROPERTIES
    PREFIX ""
    LIBRARY DESTINATION
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/prc
)

# Install the include headers
install(DIRECTORY include/ DESTINATION include/)

# Install the Python package
ament_python_install_package(prc)


# Install the wrapper library into the Python site-packages directory
install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
)
install(TARGETS ${PROJECT_NAME}_wrapper_cpp
    LIBRARY DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages
)

# Export package dependencies for ROS 2
ament_export_dependencies(ament_cmake Eigen3 Boost eigenpy ament_cmake_python)


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
