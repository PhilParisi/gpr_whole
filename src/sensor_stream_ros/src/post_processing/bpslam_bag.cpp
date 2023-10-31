#include "../../include/sensor_stream_ros/bpslam/bpslam_bag_processor.h"


int main(int argc, char **argv){
  ros::init(argc, argv, "bpslam_bag");
  ss::bpslam::BagProcessor bpslam;
  bpslam.getPfConfig().ekf_params->random_vars.push_back(ss::idx::x_linear);
  bpslam.getPfConfig().ekf_params->random_vars.push_back(ss::idx::y_linear);
  //bpslam.getPfConfig().ekf_params->random_vars.push_back(ss::idx::z_linear);
  bpslam.loadURDF(argv[2]);
  bpslam.openBag(argv[1]);
  bpslam.readBag();
}
