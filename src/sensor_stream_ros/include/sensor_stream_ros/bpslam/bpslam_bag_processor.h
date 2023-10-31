#ifndef BPSLAMBAGPROCESSOR_H
#define BPSLAMBAGPROCESSOR_H

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <boost/filesystem.hpp>
#include <stdexcept>
#include <map>
#include <unordered_set>
#include "bpslam.h"

namespace ss{ namespace bpslam {


class BagProcessor
{
public:
  typedef std::shared_ptr<BagProcessor> Ptr;
  BagProcessor();
  void loadURDF(std::string urdf_filename){pf_->addURDF(urdf_filename);}
  void openBag(boost::filesystem::path filename);
  void readBag();
  void spinOnce();
  bool readNext();
  void spinFilter();
  void spin();
  BPSLAMConfig & getPfConfig(){return pf_->config_;}
  BPSlam::Ptr pf_;
private:
  rosbag::Bag bag_;
  std::map<std::string , std::unordered_set<std::string>> bag_topics_;
  rosbag::View::iterator view_itterator_;
  std::shared_ptr<rosbag::View> view_;
};

}}

#endif // BPSLAMBAGPROCESSOR_H
