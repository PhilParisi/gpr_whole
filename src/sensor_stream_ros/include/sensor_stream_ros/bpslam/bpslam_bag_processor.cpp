#include "bpslam_bag_processor.h"

namespace ss{ namespace bpslam {

BagProcessor::BagProcessor()
{
  pf_.reset(new BPSlam);
}

void BagProcessor::openBag(boost::filesystem::path filename){
  if(!boost::filesystem::exists(filename)){
    std::string error = filename.string()+" does not exist";
    throw std::runtime_error(error);
  }

  std::cout << "opening bag...";
  bag_.open(filename.string());
  std::cout << " done" << std::endl;

  std::cout << "searching for topics...";
  rosbag::View topic_view(bag_);
  for (const rosbag::ConnectionInfo* info: topic_view.getConnections()) {
    bag_topics_[info->datatype].insert(info->topic);
  }
  std::cout << " done" << std::endl;

  return;
}

void BagProcessor::readBag(){
  std::vector<std::string> topics;
  topics.push_back(*bag_topics_["sensor_msgs/PointCloud2"].begin());
  topics.push_back(std::string("/nav/processed/odometry"));
  topics.push_back(std::string("odometry/filtered"));
  view_.reset(new rosbag::View(bag_,rosbag::TopicQuery(topics)));
  view_itterator_=view_->begin();
}

void BagProcessor::spinOnce(){
  // add the next message
  readNext();
  // spin the filter
  spinFilter();
}

bool BagProcessor::readNext(){
  if(view_itterator_!=view_->end()){
    rosbag::MessageInstance const & m = *view_itterator_;
    nav_msgs::Odometry::ConstPtr odom_msg = m.instantiate<nav_msgs::Odometry>();
    if(odom_msg){
      pf_->addOdometry(odom_msg);
    }
    sensor_msgs::PointCloud2::ConstPtr cloud_msg = m.instantiate<sensor_msgs::PointCloud2>();
    if(cloud_msg){
      pf_->addSensorPing(cloud_msg);
    }
    view_itterator_++;
    return true;
  }else {
    return false;
  }

}

void BagProcessor::spinFilter(){
  pf_->spinOnce();
}

void BagProcessor::spin(){
  while(view_itterator_!=view_->end()){
    spinOnce();
  }
}



}}
