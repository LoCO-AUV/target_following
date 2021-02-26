cmake_minimum_required(VERSION 2.8.3)
project(target_following)


find_package(OpenCV REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  std_srvs
  tf
  image_transport
  image_geometry
  message_filters
  cv_bridge
  dynamic_reconfigure
  message_generation
)



add_message_files(
   FILES
   TargetObservation.msg
   TargetObservations.msg 
   Params.msg
)

generate_messages(
   DEPENDENCIES
   geometry_msgs   sensor_msgs   std_msgs
 )

generate_dynamic_reconfigure_options(
  cfg/BBoxControllerParams.cfg
  cfg/PIDControllerParams.cfg
  cfg/ColorBBoxParams.cfg
  cfg/DRPControllerParams.cfg
)

catkin_package(
#  LIBRARIES target_following
   CATKIN_DEPENDS geometry_msgs roscpp rospy sensor_msgs std_msgs tf message_runtime
#  DEPENDS system_lib
)


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)
