#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <vector>
#include "../include/sensor_stream_ros/gpr_mapper.h"
#include "../include/sensor_stream_ros/gpr_bag_mapper.h"
#include "yaml-cpp/yaml.h"
#include <boost/filesystem.hpp>

//struct GprAnalytics{
//  std::string test_name;
//  double total_time;
//  std::vector<double> tile_time;
//  std::vector<size_t> tile_mem_usage;
//  boost::filesystem::path directory;
//  std::vector<std::string> images;
//  GprMapperConfig config;
//  GprAnalytics(){
//    return;
//  }
//  void write(){

//    YAML::Node analytics;
//    analytics["total_time"] = total_time;
//    analytics["tile_time"] = tile_time;
//    analytics["tile_mem_usage"] = tile_mem_usage;

//    boost::filesystem::path outdir(directory.string()+"/"+test_name);
//    if(!boost::filesystem::exists(outdir)){
//      boost::filesystem::create_directory(outdir);
//    }
//    std::string filename = outdir.string()+"/analytics.yml";
//    std::ofstream fout(filename);
//    fout << analytics;
//  }

//  void writeConfig(){
//    boost::filesystem::path outdir(directory.string()+"/"+test_name);
//    if(!boost::filesystem::exists(outdir)){
//      boost::filesystem::create_directory(outdir);
//    }
//    std::string filename = outdir.string()+"/config.yml";
//    config.writeToYaml(filename);
//  }
//  void addSparsityMat(cv::Mat sparsity){
//    boost::filesystem::path filename(directory.string()+"/"+test_name+"/sparsity/img_"+std::to_string(images.size())+".jpg");
//    //std::string root = filename.parent_path().string();
//    if(!boost::filesystem::exists(filename.parent_path())){
//      boost::filesystem::create_directory(filename.parent_path());
//    }
//    images.push_back(filename.filename().string());
//    cv::imwrite( filename.string(), sparsity );
//  }

//};

//class GprBagMapper{
//public:
//  GprBagMapper(std::shared_ptr<rosbag::Bag> bag){
//    bagPtr=bag;
//    mapper.cfg.inCloudTopic = "/detections";
//    topics.push_back(std::string(mapper.cfg.inCloudTopic));
//    topics.push_back(std::string("/tf"));
//  }

//  void run(){
//    for(rosbag::MessageInstance const m: rosbag::View(*bagPtr,rosbag::TopicQuery(topics)))
//    {
//      sensor_msgs::PointCloud2::ConstPtr detection = m.instantiate<sensor_msgs::PointCloud2>();
//      if (detection){
//        mapper.inCloudCallback(detection);
//      }

//      tf2_msgs::TFMessage::ConstPtr transform = m.instantiate<tf2_msgs::TFMessage>();
//      if(transform){
//        for(size_t i =0 ; i<transform->transforms.size();i++){
//          mapper._tfBuffer.setTransform(transform->transforms[i],"patch_tester");
//        }
//      }
//    }
//    double totalTime=0;
//    size_t free_baseline,total_free_baseline, base_usage;
//    cudaMemGetInfo(&free_baseline,&total_free_baseline);
//    base_usage= total_free_baseline-free_baseline;


//    //analytics.directory="/home/kris/Data/testing/bag";
//    //analytics.test_name="experiment1";
//    analytics.config=mapper.cfg;
//    //mapper.cfg=analytics.config;
//    analytics.writeConfig();
//    while(mapper.tileQueueHasElements()){
//      size_t free,total;
//      cudaMemGetInfo(&free,&total);
//      ros::Time start = ros::Time::now();
//      mapper.spinOnce();
//      ros::Time end = ros::Time::now();
//      ros::Duration duration = end-start;
//      totalTime+=duration.toSec();
//      ROS_INFO("   time: %f",duration.toSec());
//      ROS_INFO("   used: %f Mb",double(total-free-base_usage)/1e6);


//      analytics.tile_time.push_back(duration.toSec());
//      analytics.tile_mem_usage.push_back(total-free-base_usage);
//      cv::Mat sparsity = ss::viz::sparsity2image(mapper.gpr->getCholeskyFactor(),2.0f*float(mapper.cfg.block->size/100));
//      cv::imshow( "SparsityPattern",
//                  sparsity
//                  );
//      cv::waitKey(100);
//      analytics.addSparsityMat(
//            sparsity
//            );
//      analytics.total_time=totalTime;
//      analytics.write();
//    }
//    ROS_INFO("Total Time: %f",totalTime);

//  }

//  GprMapper mapper;
//  std::vector<std::string> topics;
//  std::shared_ptr<rosbag::Bag> bagPtr;
//  GprAnalytics analytics;

//};


int main(int argc, char **argv)
{
  ros::init(argc, argv, "gpr_bagmapper_node");
  std::shared_ptr<rosbag::Bag> bag;
  bag.reset(new rosbag::Bag);
  if(argc < 2){
    std::cerr << "ERROR: You must input a bag file" << std::endl;
    return 1;
  }
  if(argc < 3){
    std::cerr << "ERROR: You must input an output directory" << std::endl;
    return 1;
  }
  bag->open(argv[1]);

//  {
//    GprBagMapper mapper(bag);
//    mapper.analytics.directory=argv[2];
//    mapper.analytics.test_name="block_size_0200";
//    mapper.mapper.cfg.block->size=200;
//    mapper.run();
//  }
  {
    GprBagMapper mapper(bag);
    mapper.analytics.directory=argv[2];
    mapper.analytics.test_name="block_size_800";
    mapper.mapper.cfg.block->size=800;
    mapper.run();
  }
//  {
//    GprBagMapper mapper(bag);
//    mapper.analytics.directory=argv[2];
//    mapper.analytics.test_name="block_size_1600";
//    mapper.mapper.cfg.block->size=1600;
//    mapper.run();
//  }
////  {
////    GprBagMapper mapper(bag);
////    mapper.analytics.directory="/data/testing/SensorStream/block_size_tests";
////    mapper.analytics.test_name="block_size_2400";
////    mapper.mapper.cfg.block->size=2400;
////    mapper.run();
////  }
}
