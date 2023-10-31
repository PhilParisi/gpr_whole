#cmake_minimum_required(VERSION 2.8.3)

#get_filename_component(PACKAGE_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
#message("Creating Package: ${PACKAGE_NAME}")

#project(${PACKAGE_NAME})  ## this package name is the name of the directory this cmake file is in


#################################
###  CONFIGURE THIS FILE HERE  ##
#################################

#set(ROS_DEPENDS
#  #${PROJECT_NAME}_msgs
#  multibeam_process_core
#  roscpp
#  std_msgs
#  laser_geometry
#  pcl_conversions
#  pcl_ros
#  visualization_msgs
#  tf2
#  tf2_ros
#  tf2_msgs
#  tf2_sensor_msgs
#  geometry_msgs
#  sensor_msgs
#  nav_msgs
#  urdf
#  )

## if you end up using external dependencies it be a bit more manual (it's not that bad don't worry)
#set(EXTERNAL_DEPENDS
#  octomap
#  yaml-cpp
#  )

#set(EXTERNAL_DEPENDS_LIB
#  ${OCTOMAP_LIBRARIES}
#  yaml-cpp
#  Qt5::Widgets  Qt5::PrintSupport Qt5::Core
#  )


###########################################
###  Be Carefull editing below this line ##
###########################################

### Compile as C++11, supported in ROS Kinetic and newer
#add_compile_options(-std=c++11)

####
####  Add Non ROS dependencies
####

#foreach(DEPEND ${EXTERNAL_DEPENDS})
#  find_package(${DEPEND} REQUIRED)
#endforeach(DEPEND)

####
####  Add any required ros packages here
####
#find_package(catkin REQUIRED COMPONENTS ${ROS_DEPENDS})

#catkin_package(
#  INCLUDE_DIRS
#    include
#  LIBRARIES
#    ${PROJECT_NAME}
#  CATKIN_DEPENDS
#    ${ROS_DEPENDS}
#  DEPENDS
#    ${EXTERNAL_DEPENDS}
#)

### Specify additional locations of header files
### Your package locations should be listed before other locations
#include_directories(
#  include
#  ${catkin_INCLUDE_DIRS}
#)

#message("catkin_include_dirs" )
#foreach(source ${catkin_INCLUDE_DIRS})
#  message("  - " ${source})
#endforeach(source)

####
#### QT stuff (required if you want a GUI)
####
#set(CMAKE_AUTOMOC ON)  #This will autogenerate qt gui stuff
#set(CMAKE_AUTORCC ON)  #this needs to be on so cmake will know what to do with QT resource Files

#find_package(Qt5Widgets)
#find_package(Qt5PrintSupport)
##find all the qt UI stuff
#file(GLOB_RECURSE UI
#    "include/${PROJECT_NAME}/*.ui"
#)

#message("Adding ${PROJECT_NAME} UI files: " )
#foreach(source ${UI})
#    message("  - " ${source})
#endforeach(source)

##make them into headers
#qt5_wrap_ui (UI_HDRS  ${UI})

#message("Generating ${PROJECT_NAME} UI He1aders: " )
#foreach(source ${UI_HDRS})
#    message("  - " ${source})
#endforeach(source)

############
### Build ##
############

#file(GLOB_RECURSE HDRS
#  "include/${PROJECT_NAME}/*.h"
#)
#file(GLOB_RECURSE SRCS
#  "src/*.cpp"
#  "src/*.hpp"
#)
#file(GLOB NODES
#  "nodes/*.cpp"
#)

#message("buliding ${PROJECT_NAME} node: ")
#foreach(NODE_FILE ${NODES})
#  get_filename_component(NODE_NAME ${NODE_FILE} NAME_WE)
#  message("  - " ${NODE_NAME})

#  add_executable(${NODE_NAME} ${NODE_FILE} ${UI_HDRS} ${HDRS} ${SRCS} )
#  #add_dependencies(${NODE_NAME} ${PROJECT_NAME}_msgs_generate_messages_cpp)   # this makes sure required messages are addede before compilation
#  target_link_libraries (${NODE_NAME} ${catkin_LIBRARIES} ${EXTERNAL_DEPENDS_LIB})
#  target_include_directories(${NODE_NAME} PRIVATE "include/${PROJECT_NAME}/")
#  #target_include_directories(${NODE_NAME} PRIVATE "/home/kris/pos_ws/build/multibeam_process/multibeam_process_tools/")

#endforeach(NODE_FILE)




###
###  This is the stuff that allows your project to be used as a library for another project
###
#add_library(${PROJECT_NAME}
#    ${HDRS} ${SRCS} ${UI_HDRS}
#)
#add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} )
#target_link_libraries (${PROJECT_NAME} ${catkin_LIBRARIES} ${EXTERNAL_DEPENDS_LIB})
#target_include_directories(${PROJECT_NAME} PRIVATE "include/${PROJECT_NAME}/")

#install(DIRECTORY include/${PROJECT_NAME}/
#    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#    FILES_MATCHING PATTERN "*.h"
#    PATTERN ".svn" EXCLUDE
#)

#install(TARGETS ${PROJECT_NAME}
#    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#)









cmake_minimum_required(VERSION 2.8.3)
project(multibeam_process_tools)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++14)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(catkin REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(octomap REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  multibeam_process_core
  roscpp
  std_msgs
  octomap_ros
  laser_geometry
  pcl_conversions
  pcl_ros
  visualization_msgs
  tf2
  tf2_ros
  tf2_msgs
  tf2_sensor_msgs
  geometry_msgs
  sensor_msgs
  nav_msgs
  urdf
#  grid_map_core
#  grid_map_ros
#  grid_map_cv
#  grid_map_filters
#  grid_map_loader
#  grid_map_msgs
#  grid_map_octomap
#  grid_map_rviz_plugin
#  grid_map_visualization

)


catkin_package(
  INCLUDE_DIRS include  ### this line is important for including this package in other catkin packages
  LIBRARIES ${PROJECT_NAME}
  DEPENDS
     roscpp
     laser_geometry
     pcl_conversions
     pcl_ros
     visualization_msgs
     tf2
     tf2_ros
     tf2_msgs
     tf2_sensor_msgs
     sensor_msgs
     nav_msgs
)




include_directories(${catkin_INCLUDE_DIRS})
include_directories(${roscpp_INCLUDE_DIRS})
include_directories(${std_msgs_INCLUDE_DIRS})
include_directories(${OCTOMAP_INCLUDE_DIRS})


###
### QT stuff (required if you want a GUI)
###
set(CMAKE_AUTOMOC ON)  #This will autogenerate qt gui stuff
set(CMAKE_AUTORCC ON)  #this needs to be on so cmake will know what to do with QT resource Files

find_package(Qt5Widgets)
find_package(Qt5PrintSupport)
#find all the qt UI stuff
file(GLOB_RECURSE UI
    "include/${PROJECT_NAME}/ui/*.ui"
)

message("Adding ${PROJECT_NAME} UI files: " )
foreach(source ${UI})
    message("  - " ${source})
endforeach(source)

#make them into headers
qt5_wrap_ui (MB_UI_HDRS  ${UI})

message("Generating multibeam_process UI Headers: " )
foreach(source ${MB_UI_HDRS})
    message("  - " ${source})
endforeach(source)

###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

file(GLOB_RECURSE SRCS
    "include/${PROJECT_NAME}/*.h"
    "include/${PROJECT_NAME}/*.cpp"
    "include/${PROJECT_NAME}/*.hpp"
    "src/*.cpp"
    "src/*.hpp"
)

message("Generating multibeam_process srcs: " )
foreach(source ${SRCS})
    message("  - " ${source})
endforeach(source)


add_executable(patch_tester nodes/patch_tester.cpp ${MB_UI_HDRS} ${SRCS} )
target_link_libraries(patch_tester yaml-cpp Qt5::Widgets  Qt5::PrintSupport Qt5::Core ${catkin_LIBRARIES} ${PCL_LIBRARIES} )
target_include_directories(patch_tester PRIVATE "include/${PROJECT_NAME}/")

add_executable(octomap2pcd nodes/octomap2pcd.cpp)
target_link_libraries(octomap2pcd ${catkin_LIBRARIES} ${OCTOMAP_LIBRARIES} )

