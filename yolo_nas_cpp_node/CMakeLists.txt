cmake_minimum_required(VERSION 3.8)
project(yolo_nas_cpp_node)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

if(NOT yolo_nas_cpp_FOUND)
  message(WARNING "skipping yolo_nas_cpp_node, no yolo_nas_cpp found")
  return()
endif()

ament_auto_add_library(yolo_nas_cpp_node SHARED
  src/yolo_nas_cpp_node.cpp
)
rclcpp_components_register_node(
  yolo_nas_cpp_node
  PLUGIN "yolo_nas_cpp_node::YoloNasNode"
  EXECUTABLE yolo_nas_cpp_node_exec
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(
  INSTALL_TO_SHARE
  launch
)