cmake_minimum_required(VERSION 3.5.1)
set(CUDA_TOOLKIT_ROOT_DIR "/usr/lib/cuda")
find_package(CUDA REQUIRED)

add_subdirectory(sensor_stream)


find_package(Ceres REQUIRED)

project(sensor_stream_ros)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++14)
## need this for qt auto generated headers to include!
set(CMAKE_INCLUDE_CURRENT_DIR ON)


## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_library(CUDA_NVTX_LIBRARY
  NAMES nvToolsExt nvTools nvtoolsext nvtools nvtx NVTX
  PATHS ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES "lib64" "common/lib64" "common/lib" "lib"
  DOC "Location of the CUDA Toolkit Extension (NVTX) library"
  NO_DEFAULT_PATH
  )


find_package(OpenCV REQUIRED)
find_package(  yaml-cpp REQUIRED )
find_package(Boost REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
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
  image_transport
  cv_bridge
  multibeam_process_core
  grid_map_core
  grid_map_ros
  grid_map_cv
  grid_map_filters
  grid_map_loader
  grid_map_msgs
  grid_map_octomap
  grid_map_rviz_plugin
  grid_map_visualization
)


catkin_package(
  INCLUDE_DIRS include  ### this line is important for including this package in other catkin packages
  LIBRARIES ${PROJECT_NAME}
  DEPENDS
     roscpp
     pcl_conversions
     pcl_ros
     visualization_msgs
     tf2_sensor_msgs
     sensor_msgs
     nav_msgs
     multibeam_process
)

include_directories(${catkin_INCLUDE_DIRS})
include_directories(${roscpp_INCLUDE_DIRS})
include_directories(${std_msgs_INCLUDE_DIRS})

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
qt5_wrap_ui (UI_HDRS  ${UI})

message("Generating ${PROJECT_NAME} UI Headers: " )
foreach(source ${UI_HDRS})
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
    "include/${PROJECT_NAME}/*.cu"
    "include/${PROJECT_NAME}/*.hpp"
)

message("Adding ${PROJECT_NAME} SRC files: " )
foreach(source ${SRCS})
    message("  - " ${source})
endforeach(source)



#add_library(${PROJECT_NAME}
#    ${SRCS} ${UI_HDRS}
#)
#target_link_libraries(${PROJECT_NAME}
#  sensor_stream yaml-cpp ceres
#  ${catkin_LIBRARIES} ${CUDA_NVTX_LIBRARY} ${OpenCV_LIBS}
#  Qt5::Widgets  Qt5::PrintSupport Qt5::Core)

#add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} sensor_stream yaml-cpp ceres
#                 Qt5::Widgets  Qt5::PrintSupport Qt5::Core)


#install(DIRECTORY include/${PROJECT_NAME}/
#    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#    FILES_MATCHING PATTERN "*.h*"
#    PATTERN ".svn" EXCLUDE
#)


#install(TARGETS ${PROJECT_NAME}
#    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#    LIBRARY DESTIsensor_stream_ros/ui/bpslam_particle_widget.hNATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#)


function(add_directory DIRECTORY)

  file(GLOB files
      "src/${DIRECTORY}/*.cpp"
  )
  message("adding ${DIRECTORY}: ")
  foreach(source ${files})
    get_filename_component(main_cpp ${source} NAME_WE)
    message("  - " ${main_cpp})
    cuda_add_executable(${main_cpp} ${source} ${SRCS} ${UI_HDRS}
        OPTIONS -std=c++11 -Xcompiler -rdynamic -lineinfo --default-stream per-thread)
    target_link_libraries(${main_cpp} sensor_stream yaml-cpp ceres ${catkin_LIBRARIES}
                            ${CUDA_NVTX_LIBRARY} ${OpenCV_LIBS}
                            Qt5::Widgets  Qt5::PrintSupport Qt5::Core)
#    add_executable(${main_cpp} ${source} ${SRCS} ${UI_HDRS})
#    target_link_libraries(${main_cpp} sensor_stream yaml-cpp ceres ${catkin_LIBRARIES}
#                          ${CUDA_LIBRARIES} ${CUDA_NVTX_LIBRARY} ${OpenCV_LIBS}
#                          Qt5::Widgets  Qt5::PrintSupport Qt5::Core)
  endforeach(source)

endfunction(add_directory)

#add_directory("experiments")
#add_directory("nodes")
add_directory("post_processing")
#add_directory("scratch")


file(GLOB tests
    "tests/*.cpp"
)
message("adding tests: ")
foreach(source ${tests})
  get_filename_component(main_cpp ${source} NAME_WE)
  message("  - " ${main_cpp})
  catkin_add_gtest(${main_cpp} ${source} ${SRCS} ${UI_HDRS})
  if(TARGET ${main_cpp})
    target_link_libraries(${main_cpp} sensor_stream yaml-cpp ceres ${catkin_LIBRARIES}
      ${CUDA_NVTX_LIBRARY} ${OpenCV_LIBS}
      Qt5::Widgets  Qt5::PrintSupport Qt5::Core)
  endif()
endforeach(source)


