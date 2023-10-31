#include <ros/ros.h>
#include "../include/sensor_stream_ros/gpr_mapper.h"

int main(int argc, char **argv)
{

  ros::init(argc, argv, "gpr_mapper_node");
  GprMapper test;
  test.spin();
}
