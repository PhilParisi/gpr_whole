#include "gridmapper.h"

int main(int argc, char **argv)
{
  ros::init(argc, argv, "gridmapper_node");

  GridMapper mapper;
  mapper.spin();

  return 0;
}
