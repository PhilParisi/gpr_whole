#include "octomapper.h"

int main(int argc, char **argv)
{
  ros::init(argc, argv, "octomapper_node");

  Octomapper mapper;
  mapper.spin();

  return 0;
}
