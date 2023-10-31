#ifndef GPR_BAG_MAPPER_H
#define GPR_BAG_MAPPER_H

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <vector>
#include "../include/sensor_stream_ros/gpr_mapper.h"
#include "yaml-cpp/yaml.h"
#include <boost/filesystem.hpp>


struct GprAnalytics{
  std::string test_name;
  double total_time;
  std::vector<double> tile_time;
  std::vector<double> tile_mem_usage;
  std::vector<int> tile_nnz;
  std::vector<int> tile_chol_rows;
  boost::filesystem::path directory;
  std::vector<std::string> images;
  std::vector<std::string> tiles;
  GprMapperConfig config;
  GprAnalytics(){
    return;
  }
  void write();
  void writeConfig();
  void addSparsityMat(cv::Mat sparsity);
  void addPrediction(ss::ros::Tile working_tile);

};

struct HyperparamAnalytics{
  std::string test_name;
  boost::filesystem::path directory;
  std::vector<float> sigma2e;
  std::vector<std::vector<float> > hyperparams;
  std::vector<float> lml;
  YAML::Node asYaml();
  void write();
};

class GprBagMapper{
public:
  GprBagMapper(std::shared_ptr<rosbag::Bag> bag);
  void processBag();
  void run();

  GprMapper mapper;
  std::vector<std::string> topics;
  std::shared_ptr<rosbag::Bag> bagPtr;
  GprAnalytics analytics;

};
#endif // GPR_BAG_MAPPER_H
