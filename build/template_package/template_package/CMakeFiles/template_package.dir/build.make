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

# Include any dependencies generated for this target.
include template_package/template_package/CMakeFiles/template_package.dir/depend.make

# Include the progress variables for this target.
include template_package/template_package/CMakeFiles/template_package.dir/progress.make

# Include the compile flags for this target's objects.
include template_package/template_package/CMakeFiles/template_package.dir/flags.make

template_package/template_package/CMakeFiles/template_package.dir/src/template_class.cpp.o: template_package/template_package/CMakeFiles/template_package.dir/flags.make
template_package/template_package/CMakeFiles/template_package.dir/src/template_class.cpp.o: /home/parisi/Desktop/catkin_ws/src/template_package/template_package/src/template_class.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/parisi/Desktop/catkin_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object template_package/template_package/CMakeFiles/template_package.dir/src/template_class.cpp.o"
	cd /home/parisi/Desktop/catkin_ws/build/template_package/template_package && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/template_package.dir/src/template_class.cpp.o -c /home/parisi/Desktop/catkin_ws/src/template_package/template_package/src/template_class.cpp

template_package/template_package/CMakeFiles/template_package.dir/src/template_class.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/template_package.dir/src/template_class.cpp.i"
	cd /home/parisi/Desktop/catkin_ws/build/template_package/template_package && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/parisi/Desktop/catkin_ws/src/template_package/template_package/src/template_class.cpp > CMakeFiles/template_package.dir/src/template_class.cpp.i

template_package/template_package/CMakeFiles/template_package.dir/src/template_class.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/template_package.dir/src/template_class.cpp.s"
	cd /home/parisi/Desktop/catkin_ws/build/template_package/template_package && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/parisi/Desktop/catkin_ws/src/template_package/template_package/src/template_class.cpp -o CMakeFiles/template_package.dir/src/template_class.cpp.s

# Object files for target template_package
template_package_OBJECTS = \
"CMakeFiles/template_package.dir/src/template_class.cpp.o"

# External object files for target template_package
template_package_EXTERNAL_OBJECTS =

/home/parisi/Desktop/catkin_ws/devel/lib/libtemplate_package.so: template_package/template_package/CMakeFiles/template_package.dir/src/template_class.cpp.o
/home/parisi/Desktop/catkin_ws/devel/lib/libtemplate_package.so: template_package/template_package/CMakeFiles/template_package.dir/build.make
/home/parisi/Desktop/catkin_ws/devel/lib/libtemplate_package.so: template_package/template_package/CMakeFiles/template_package.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/parisi/Desktop/catkin_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library /home/parisi/Desktop/catkin_ws/devel/lib/libtemplate_package.so"
	cd /home/parisi/Desktop/catkin_ws/build/template_package/template_package && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/template_package.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
template_package/template_package/CMakeFiles/template_package.dir/build: /home/parisi/Desktop/catkin_ws/devel/lib/libtemplate_package.so

.PHONY : template_package/template_package/CMakeFiles/template_package.dir/build

template_package/template_package/CMakeFiles/template_package.dir/clean:
	cd /home/parisi/Desktop/catkin_ws/build/template_package/template_package && $(CMAKE_COMMAND) -P CMakeFiles/template_package.dir/cmake_clean.cmake
.PHONY : template_package/template_package/CMakeFiles/template_package.dir/clean

template_package/template_package/CMakeFiles/template_package.dir/depend:
	cd /home/parisi/Desktop/catkin_ws/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/parisi/Desktop/catkin_ws/src /home/parisi/Desktop/catkin_ws/src/template_package/template_package /home/parisi/Desktop/catkin_ws/build /home/parisi/Desktop/catkin_ws/build/template_package/template_package /home/parisi/Desktop/catkin_ws/build/template_package/template_package/CMakeFiles/template_package.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : template_package/template_package/CMakeFiles/template_package.dir/depend

