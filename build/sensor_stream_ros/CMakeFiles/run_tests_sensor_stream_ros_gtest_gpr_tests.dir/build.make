# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/parisi/Desktop/catkin_ws/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/parisi/Desktop/catkin_ws/build

# Utility rule file for run_tests_sensor_stream_ros_gtest_gpr_tests.

# Include the progress variables for this target.
include sensor_stream_ros/CMakeFiles/run_tests_sensor_stream_ros_gtest_gpr_tests.dir/progress.make

sensor_stream_ros/CMakeFiles/run_tests_sensor_stream_ros_gtest_gpr_tests:
	cd /home/parisi/Desktop/catkin_ws/build/sensor_stream_ros && ../catkin_generated/env_cached.sh /usr/bin/python3 /opt/ros/noetic/share/catkin/cmake/test/run_tests.py /home/parisi/Desktop/catkin_ws/build/test_results/sensor_stream_ros/gtest-gpr_tests.xml "/home/parisi/Desktop/catkin_ws/devel/lib/sensor_stream_ros/gpr_tests --gtest_output=xml:/home/parisi/Desktop/catkin_ws/build/test_results/sensor_stream_ros/gtest-gpr_tests.xml"

run_tests_sensor_stream_ros_gtest_gpr_tests: sensor_stream_ros/CMakeFiles/run_tests_sensor_stream_ros_gtest_gpr_tests
run_tests_sensor_stream_ros_gtest_gpr_tests: sensor_stream_ros/CMakeFiles/run_tests_sensor_stream_ros_gtest_gpr_tests.dir/build.make

.PHONY : run_tests_sensor_stream_ros_gtest_gpr_tests

# Rule to build all files generated by this target.
sensor_stream_ros/CMakeFiles/run_tests_sensor_stream_ros_gtest_gpr_tests.dir/build: run_tests_sensor_stream_ros_gtest_gpr_tests

.PHONY : sensor_stream_ros/CMakeFiles/run_tests_sensor_stream_ros_gtest_gpr_tests.dir/build

sensor_stream_ros/CMakeFiles/run_tests_sensor_stream_ros_gtest_gpr_tests.dir/clean:
	cd /home/parisi/Desktop/catkin_ws/build/sensor_stream_ros && $(CMAKE_COMMAND) -P CMakeFiles/run_tests_sensor_stream_ros_gtest_gpr_tests.dir/cmake_clean.cmake
.PHONY : sensor_stream_ros/CMakeFiles/run_tests_sensor_stream_ros_gtest_gpr_tests.dir/clean

sensor_stream_ros/CMakeFiles/run_tests_sensor_stream_ros_gtest_gpr_tests.dir/depend:
	cd /home/parisi/Desktop/catkin_ws/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/parisi/Desktop/catkin_ws/src /home/parisi/Desktop/catkin_ws/src/sensor_stream_ros /home/parisi/Desktop/catkin_ws/build /home/parisi/Desktop/catkin_ws/build/sensor_stream_ros /home/parisi/Desktop/catkin_ws/build/sensor_stream_ros/CMakeFiles/run_tests_sensor_stream_ros_gtest_gpr_tests.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : sensor_stream_ros/CMakeFiles/run_tests_sensor_stream_ros_gtest_gpr_tests.dir/depend

