cmake_minimum_required(VERSION 2.8.3)
project(object_reconstruction)

find_package(catkin REQUIRED COMPONENTS
  dynamic_reconfigure
)

generate_dynamic_reconfigure_options(
  cfg/TransformPointcloud.cfg
)

find_package(catkin REQUIRED COMPONENTS rospy)
