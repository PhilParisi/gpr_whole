#include "gpr_bag_mapper.h"

void GprAnalytics::write(){
  YAML::Node analytics;
  analytics["total_time"] = total_time;
  analytics["tile_time"] = tile_time;
  analytics["tile_mem_usage"] = tile_mem_usage;
  analytics["tile_nnz"]=tile_nnz;
  analytics["tile_chol_rows"]=tile_chol_rows;

  boost::filesystem::path outdir(directory.string()+"/"+test_name);
  if(!boost::filesystem::exists(outdir)){
    boost::filesystem::create_directory(outdir);
  }
  std::string filename = outdir.string()+"/analytics.yml";
  std::ofstream fout(filename);
  fout << analytics;
}

void GprAnalytics::writeConfig(){
  boost::filesystem::path outdir(directory.string()+"/"+test_name);
  if(!boost::filesystem::exists(outdir)){
    boost::filesystem::create_directory(outdir);
  }
  std::string filename = outdir.string()+"/config.yml";
  config.writeToYaml(filename);
}

void GprAnalytics::addSparsityMat(cv::Mat sparsity){
  boost::filesystem::path filename(directory.string()+"/"+test_name+"/sparsity/img_"+std::to_string(images.size())+".jpg");
  //std::string root = filename.parent_path().string();
  if(!boost::filesystem::exists(filename.parent_path())){
    boost::filesystem::create_directory(filename.parent_path());
  }
  images.push_back(filename.filename().string());
  cv::imwrite( filename.string(), sparsity );
}

void GprAnalytics::addPrediction(ss::ros::Tile working_tile){
  boost::filesystem::path filename(directory.string()+"/"+test_name+"/predictions/cloud_"+std::to_string(tiles.size())+".pcd");
  //std::string root = filename.parent_path().string();
  if(!boost::filesystem::exists(filename.parent_path())){
    boost::filesystem::create_directory(filename.parent_path());
  }
  tiles.push_back(filename.filename().string());

  auto cloud_msg = working_tile.getPrediction();
  pcl::PointCloud<pcl::PointXYZI> pcl_cloud;
  pcl::fromROSMsg(*cloud_msg,pcl_cloud);
  pcl::io::savePCDFileBinary (filename.string(), pcl_cloud);


}

YAML::Node HyperparamAnalytics::asYaml(){
  YAML::Node node;
  node["test_name"] = test_name;
  node["sigma2e"] = sigma2e;
  node["hyperparams"] = hyperparams;
  node["lml"] = lml;
  return node;
}

void HyperparamAnalytics::write(){
  boost::filesystem::path outdir(directory.string()+"/"+test_name);
  if(!boost::filesystem::exists(outdir)){
    boost::filesystem::create_directory(outdir);
  }
  std::string filename = outdir.string()+"/hyperparam_analytics.yml";
  std::ofstream fout(filename);
  fout << asYaml();
}




//
// GprBagMapper
//

GprBagMapper::GprBagMapper(std::shared_ptr<rosbag::Bag> bag){
  {
      bagPtr=bag;
      mapper.cfg.inCloudTopic = "/detections";
      topics.push_back(std::string(mapper.cfg.inCloudTopic));
      topics.push_back(std::string("/tf"));
    }
}

void GprBagMapper::processBag(){
  for(rosbag::MessageInstance const m: rosbag::View(*bagPtr,rosbag::TopicQuery(topics)))
  {
    sensor_msgs::PointCloud2::ConstPtr detection = m.instantiate<sensor_msgs::PointCloud2>();
    if (detection){
      mapper.inCloudCallback(detection);
    }

    tf2_msgs::TFMessage::ConstPtr transform = m.instantiate<tf2_msgs::TFMessage>();
    if(transform){
      for(size_t i =0 ; i<transform->transforms.size();i++){
        mapper._tfBuffer.setTransform(transform->transforms[i],"patch_tester");
      }
    }
  }
}

void GprBagMapper::run(){
  processBag();

  double totalTime=0;
  size_t free_baseline,total_free_baseline, base_usage;
  cudaMemGetInfo(&free_baseline,&total_free_baseline);
  base_usage= total_free_baseline-free_baseline;


  analytics.config=mapper.cfg;
  analytics.writeConfig();
  while(mapper.tileQueueHasElements()){
    auto working_tile = mapper.queueFront();
    size_t free,total;
    cudaMemGetInfo(&free,&total);
    ros::Time start = ros::Time::now();
    mapper.spinOnce();
    ros::Time end = ros::Time::now();
    ros::Duration duration = end-start;
    totalTime+=duration.toSec();
    ROS_INFO("   time: %f",duration.toSec());
    ROS_INFO("   used: %f Mb",(double(total-free)-double(base_usage))/1e6);


    analytics.tile_time.push_back(duration.toSec());
    analytics.tile_mem_usage.push_back((double(total-free)-double(base_usage) )/1e6);
    analytics.tile_nnz.push_back(mapper.gpr->getCholeskyFactor().nnz());
    analytics.tile_chol_rows.push_back(mapper.gpr->getCholeskyFactor().rows());
    cv::Mat sparsity = ss::viz::sparsity2image(mapper.gpr->getCholeskyFactor(),2.0f*float(mapper.cfg.block->size)/float(100));
    cv::imshow( "SparsityPattern",
                sparsity
                );
    cv::waitKey(100);
    analytics.addSparsityMat(
          sparsity
          );
    analytics.addPrediction(working_tile);
    analytics.total_time=totalTime;
    analytics.write();
  }
  ROS_INFO("Total Time: %f",totalTime);

}

