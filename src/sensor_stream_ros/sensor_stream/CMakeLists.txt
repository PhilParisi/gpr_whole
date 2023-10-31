cmake_minimum_required(VERSION 3.5.1)
project (sensor_stream)

find_package(CUDA REQUIRED)
message( "Cuda version Detected: " ${CUDA_VERSION_STRING})

find_package(PCL 1.3 REQUIRED COMPONENTS common io filters)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(CMAKE_CXX_STANDARD 11) # C++11...
set(CMAKE_CXX_STANDARD_REQUIRED ON) #...is required...
set(CMAKE_CXX_EXTENSIONS OFF) #...without compiler extensions like gnu++11


find_library(CUDA_NVTX_LIBRARY
  NAMES nvToolsExt nvTools nvtoolsext nvtools nvtx NVTX
  PATHS ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES "lib64" "common/lib64" "common/lib" "lib"
  DOC "Location of the CUDA Toolkit Extension (NVTX) library"
  NO_DEFAULT_PATH
  )

###
### adding source files
###
file(GLOB SRCS
    "include/sensor_stream/*.h"
    "include/sensor_stream/*.cu"
    "include/sensor_stream/*.cpp"
    "include/sensor_stream/gpr/*.h"
    "include/sensor_stream/gpr/*.cu"
    "include/sensor_stream/gpr/*.cpp"
)
message("Adding source files: " )
foreach(source ${SRCS})
    message("  - " ${source})
endforeach(source)

file(GLOB HDRS
    "include/sensor_stream/*.h"
    "include/sensor_stream/gpr/*.h"
)


###
### add executables
###

#cuda_add_executable (pcl_gp main/testing/pcl_gp.cpp ${SRCS} OPTIONS -std=c++11 -Xcompiler -rdynamic -lineinfo)
#target_link_libraries (pcl_gp
#    ${CUDA_cusolver_LIBRARY}
#    ${CUDA_cusparse_LIBRARY}
#    ${CUDA_cublas_LIBRARY}
#    ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES}
#)

#cuda_add_executable (pcl_tile_gp main/testing/pcl_tile_gp.cpp ${SRCS}
#    OPTIONS -std=c++11 -Xcompiler -rdynamic -lineinfo --default-stream per-thread)
#target_link_libraries (pcl_tile_gp
#    ${CUDA_cusolver_LIBRARY}
#    ${CUDA_cusparse_LIBRARY}
#    ${CUDA_cublas_LIBRARY}
#    ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_FILTERS_LIBRARIES}
#)


###
### add library
###

#add_library(sensor_stream STATIC ${SRCS})
#target_link_libraries (sensor_stream
#    ${CUDA_cusolver_LIBRARY}
#    ${CUDA_cusparse_LIBRARY}
#    ${CUDA_cublas_LIBRARY}
#)
#target_include_directories(sensor_stream PUBLIC
#  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#  $<INSTALL_INTERFACE:include>  # <prefix>/include
#)

set(LIB_NAME "sensor_stream")
cuda_add_library(sensor_stream
  ${SRCS}
  OPTIONS -std=c++11 -Xcompiler -rdynamic -lineinfo --default-stream per-thread
  RELEASE -DNDEBUG
  DEBUG -g -DDEBUG
)
target_link_libraries (sensor_stream
    ${CUDA_cusolver_LIBRARY}
    ${CUDA_cusparse_LIBRARY}
    ${CUDA_cublas_LIBRARY}
    ${CUDA_NVTX_LIBRARY}
)
target_include_directories(sensor_stream PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>  # <prefix>/include
)

set(PUBLIC_HEADERS ${HDRS})



#cuda_add_executable (gpTest main/gp_example.cpp ${SRCS} OPTIONS -std=c++11)
#target_link_libraries (gpTest
#    Qt5::Core
#    Qt5::Gui
#    Qt5::Widgets
#    Qt5::Charts
#    ${CUDA_cusolver_LIBRARY})

#CUDA_ADD_CUBLAS_TO_TARGET(gpTest)

#cuda_add_executable (gpr2d_example main/gpr2d_example.cpp ${SRCS} OPTIONS -std=c++11 -Xcompiler -rdynamic -lineinfo)
#target_link_libraries (gpr2d_example
#    Qt5::Core
#    Qt5::Gui
#    Qt5::Widgets
#    Qt5::Charts
#    Qt5::DataVisualization
#    ${CUDA_cusolver_LIBRARY}
#    ${CUDA_cusparse_LIBRARY}
#    ${CUDA_cublas_LIBRARY}
#    )

#CUDA_ADD_CUBLAS_TO_TARGET(gpr2d_example)

#cuda_add_executable (cudamat_example main/cudamat_example.cu ${SRCS} OPTIONS -std=c++11)
#target_link_libraries (cudamat_example
#    Qt5::Core
#    Qt5::Gui
#    Qt5::Widgets
#    Qt5::Charts
#    Qt5::DataVisualization
#    ${CUDA_cusolver_LIBRARY}
#    ${CUDA_cusparse_LIBRARY}
#    ${CUDA_cublas_LIBRARY}
#    ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES}
#    )

#cuda_add_executable (sparse_chol main/testing/sparse_chol.cu ${SRCS} OPTIONS -std=c++11)
#target_link_libraries (sparse_chol
#    Qt5::Core
#    Qt5::Gui
#    Qt5::Widgets
#    Qt5::Charts
#    ${CUDA_cusolver_LIBRARY}
#    ${CUDA_cusparse_LIBRARY}
#    ${CUDA_cublas_LIBRARY}
#    )

#cuda_add_executable (sparse_test main/sparse_test.cpp ${SRCS} OPTIONS -std=c++11)
#target_link_libraries (sparse_test
#    Qt5::Core
#    Qt5::Gui
#    Qt5::Widgets
#    Qt5::Charts
#    ${CUDA_cusolver_LIBRARY}
#    ${CUDA_cusparse_LIBRARY}
#    ${CUDA_cublas_LIBRARY}
#    )


#cuda_add_executable (block_gpr main/block_gpr.cpp ${SRCS} OPTIONS -std=c++11 -Xcompiler -rdynamic -lineinfo)
#target_link_libraries (block_gpr
#    Qt5::Core
#    Qt5::Gui
#    Qt5::Widgets
#    Qt5::Charts
#    Qt5::DataVisualization
#    ${CUDA_cusolver_LIBRARY}
#    ${CUDA_cusparse_LIBRARY}
#    ${CUDA_cublas_LIBRARY}
#    )

