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

# Utility rule file for _run_tests_template_package_gtest_template_package-other_test.

# Include the progress variables for this target.
include template_package/template_package/CMakeFiles/_run_tests_template_package_gtest_template_package-other_test.dir/progress.make

template_package/template_package/CMakeFiles/_run_tests_template_package_gtest_template_package-other_test:
	cd /home/parisi/Desktop/catkin_ws/build/template_package/template_package && ../../catkin_generated/env_cached.sh /usr/bin/python3 /opt/ros/noetic/share/catkin/cmake/test/run_tests.py /home/parisi/Desktop/catkin_ws/build/test_results/template_package/gtest-template_package-other_test.xml "/home/parisi/Desktop/catkin_ws/devel/lib/template_package/template_package-other_test --gtest_output=xml:/home/parisi/Desktop/catkin_ws/build/test_results/template_package/gtest-template_package-other_test.xml"

_run_tests_template_package_gtest_template_package-other_test: template_package/template_package/CMakeFiles/_run_tests_template_package_gtest_template_package-other_test
_run_tests_template_package_gtest_template_package-other_test: template_package/template_package/CMakeFiles/_run_tests_template_package_gtest_template_package-other_test.dir/build.make

.PHONY : _run_tests_template_package_gtest_template_package-other_test

# Rule to build all files generated by this target.
template_package/template_package/CMakeFiles/_run_tests_template_package_gtest_template_package-other_test.dir/build: _run_tests_template_package_gtest_template_package-other_test

.PHONY : template_package/template_package/CMakeFiles/_run_tests_template_package_gtest_template_package-other_test.dir/build

template_package/template_package/CMakeFiles/_run_tests_template_package_gtest_template_package-other_test.dir/clean:
	cd /home/parisi/Desktop/catkin_ws/build/template_package/template_package && $(CMAKE_COMMAND) -P CMakeFiles/_run_tests_template_package_gtest_template_package-other_test.dir/cmake_clean.cmake
.PHONY : template_package/template_package/CMakeFiles/_run_tests_template_package_gtest_template_package-other_test.dir/clean

template_package/template_package/CMakeFiles/_run_tests_template_package_gtest_template_package-other_test.dir/depend:
	cd /home/parisi/Desktop/catkin_ws/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/parisi/Desktop/catkin_ws/src /home/parisi/Desktop/catkin_ws/src/template_package/template_package /home/parisi/Desktop/catkin_ws/build /home/parisi/Desktop/catkin_ws/build/template_package/template_package /home/parisi/Desktop/catkin_ws/build/template_package/template_package/CMakeFiles/_run_tests_template_package_gtest_template_package-other_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : template_package/template_package/CMakeFiles/_run_tests_template_package_gtest_template_package-other_test.dir/depend

