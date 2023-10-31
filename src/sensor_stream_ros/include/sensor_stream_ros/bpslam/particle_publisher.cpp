#include "particle_publisher.h"
namespace ss{ namespace bpslam {

ParticlePublisher::ParticlePublisher()
{
  node_.reset( new(ros::NodeHandle) );
  transformed_cloud_pub_=node_->advertise<sensor_msgs::PointCloud2>("transformed_cloud",1);
  pose_pub_=node_->advertise<geometry_msgs::PoseArray>("particle_hypothesis",1);
  gpr_pub_=node_->advertise<sensor_msgs::PointCloud2>("gpr_prediction",1);
  gpr_training_pub_=node_->advertise<sensor_msgs::PointCloud2>("gpr_training_blocks",1);
  particle_border_pub_=node_->advertise<visualization_msgs::Marker>("particle_boundary", 10);
}

void ParticlePublisher::publishProjected(ParticlePtr_t particle_ptr){

  sensor_msgs::PointCloud2 msg;
  pcl::toROSMsg(*assembleCloud(particle_ptr),msg);
  msg.header.stamp=ros::Time::now();
  msg.header.frame_id = particle_ptr->getData()->getOdomFrame();
  transformed_cloud_pub_.publish(msg);
  publishOdomHypothesis(particle_ptr);
}

void ParticlePublisher::saveParticleMap(ParticlePtr_t particle_ptr, std::string filename){
  sensor_msgs::PointCloud2 msg;
  pcl::toROSMsg(*assembleCloud(particle_ptr),msg);
  msg.header.stamp=ros::Time::now();
  msg.header.frame_id = particle_ptr->getData()->getOdomFrame();
  pcl::PointCloud<pcl::PointXYZI> transformedCloud;
  pcl::fromROSMsg(msg,transformedCloud);
  pcl::io::savePCDFileBinary(filename,transformedCloud);
}

void ParticlePublisher::publishOdomHypothesis(ParticlePtr_t particle_ptr){
  geometry_msgs::PoseArray::Ptr msg;
  msg = assembleOdomHypothesis(particle_ptr);
  pose_pub_.publish(msg);
}

void ParticlePublisher::publishAncestorOdom(ParticlePtr_t particle_ptr){
  geometry_msgs::PoseArray::Ptr msg;
  msg = assembleAncestorOdom(particle_ptr);
  pose_pub_.publish(msg);
}

void ParticlePublisher::publishGprPrediction(ParticlePtr_t particle_ptr,bool recompute){
  if(particle_ptr->getData()->map.likelihood_by_point.rows()==0||recompute){
    GPRWorker worker(particle_ptr, particle_ptr->getData()->map.gpr_params);
    worker.run();
    //particle_ptr->getData()->map.gpr_params.regression->kernel->hyperparam2host();
    //std::cout << "length scale: " << particle_ptr->getData()->map.gpr_params.regression->kernel->hyperparam("length_scale") << std::endl;
  }

  size_t rows = particle_ptr->getData()->map.likelihood_by_point.rows()
      * particle_ptr->getData()->map.likelihood_by_point.getBlockParam().rows;

  pcl::PointCloud<pcl::PointXYZI>::Ptr outCloud (new pcl::PointCloud<pcl::PointXYZI>);
  outCloud->width  = rows;
  outCloud->height  = 1;
  outCloud->points.resize (rows);

  for(size_t cloud_index = 0 ; cloud_index < rows; cloud_index++){
    outCloud->points[cloud_index].x = particle_ptr->getData()->map.predicted_input.getVal(cloud_index,0);
    outCloud->points[cloud_index].y = particle_ptr->getData()->map.predicted_input.getVal(cloud_index,1);
    float likelihood = particle_ptr->getData()->map.likelihood_by_point.getVal(cloud_index,0);
    outCloud->points[cloud_index].intensity=likelihood;
    float mu = particle_ptr->getData()->map.predicted_mu.getVal(cloud_index,0);
    if(std::isnan(mu))
        outCloud->points[cloud_index].z=0;
    else
        outCloud->points[cloud_index].z=mu;
  }

  sensor_msgs::PointCloud2::Ptr cloudMsg;
  cloudMsg.reset(new sensor_msgs::PointCloud2);
  pcl::toROSMsg(*outCloud,*cloudMsg);
  cloudMsg->header.stamp = ros::Time::now();
  cloudMsg->header.frame_id = particle_ptr->getData()->getOdomFrame();
  gpr_pub_.publish(cloudMsg);
}


void ParticlePublisher::publishMeanFunction(ParticlePtr_t particle_ptr, bool recompute){
  if(particle_ptr->getData()->map.likelihood_by_point.rows()==0||recompute){
    GPRWorker worker(particle_ptr, particle_ptr->getData()->map.gpr_params);
    worker.run();
    //particle_ptr->getData()->map.gpr_params.regression->kernel->hyperparam2host();
    //std::cout << "length scale: " << particle_ptr->getData()->map.gpr_params.regression->kernel->hyperparam("length_scale") << std::endl;
  }

  auto prediciton = particle_ptr->getData()->map.mean_function->predict(particle_ptr->getData()->map.prediction_reigon.min_point.x,
                                                      particle_ptr->getData()->map.prediction_reigon.max_point.x,
                                                      particle_ptr->getData()->map.prediction_reigon.min_point.y,
                                                      particle_ptr->getData()->map.prediction_reigon.max_point.y,
                                                      30,30);

  prediciton.dev2host();
  size_t rows = prediciton.mu.rows();


  pcl::PointCloud<pcl::PointXYZI>::Ptr outCloud (new pcl::PointCloud<pcl::PointXYZI>);
  outCloud->width  = rows;
  outCloud->height  = 1;
  outCloud->points.resize (rows);

  for(size_t cloud_index = 0 ; cloud_index < rows; cloud_index++){
    outCloud->points[cloud_index].x = prediciton.points.x(cloud_index);// particle_ptr->getData()->map.predicted_input.getVal(cloud_index,0);
    outCloud->points[cloud_index].y = prediciton.points.y(cloud_index);//particle_ptr->getData()->map.predicted_input.getVal(cloud_index,1);
    //float likelihood = particle_ptr->getData()->map.likelihood_by_point.getVal(cloud_index,0);
    outCloud->points[cloud_index].intensity=prediciton.sigma.val(cloud_index);
    float mu = prediciton.mu.val(cloud_index);//particle_ptr->getData()->map.predicted_mu.getVal(cloud_index,0);
    if(std::isnan(mu))
        outCloud->points[cloud_index].z=0;
    else
        outCloud->points[cloud_index].z=mu;
  }

  sensor_msgs::PointCloud2::Ptr cloudMsg;
  cloudMsg.reset(new sensor_msgs::PointCloud2);
  pcl::toROSMsg(*outCloud,*cloudMsg);
  cloudMsg->header.stamp = ros::Time::now();
  cloudMsg->header.frame_id = particle_ptr->getData()->getOdomFrame();
  gpr_pub_.publish(cloudMsg);
}

void ParticlePublisher::PublishGPRTraining(ParticlePtr_t particle_ptr){
  pcl::PointCloud<pcl::PointXYZI> cloud;
  for(auto block : particle_ptr->getData()->map.training_blocks){
    cloud += *block->getCloud();
  }
  sensor_msgs::PointCloud2::Ptr cloudMsg;
  cloudMsg.reset(new sensor_msgs::PointCloud2);
  pcl::toROSMsg(cloud,*cloudMsg);
  cloudMsg->header.stamp = ros::Time::now();
  cloudMsg->header.frame_id = particle_ptr->getData()->getOdomFrame();
  gpr_training_pub_.publish(cloudMsg);
}

void ParticlePublisher::publishTrainingBoundary(ParticlePtr_t particle_ptr){
  visualization_msgs::Marker  line_strip;
  line_strip.header.frame_id = particle_ptr->getData()->getOdomFrame();
  line_strip.header.stamp = ros::Time::now();
  line_strip.ns = "points_and_lines";
  line_strip.action = visualization_msgs::Marker::ADD;
  line_strip.pose.orientation.w = 1.0;
  line_strip.id = 1;
  line_strip.type = visualization_msgs::Marker::LINE_STRIP;
  line_strip.scale.x = 0.1;
  line_strip.color.b = 1.0;
  line_strip.color.a = 1.0;

  {
    geometry_msgs::Point p;
    p.x = particle_ptr->getData()->map.prediction_reigon.max_point.x;
    p.y = particle_ptr->getData()->map.prediction_reigon.max_point.y;
    p.z = 0;
    line_strip.points.push_back(p);
  }

  {
    geometry_msgs::Point p;
    p.x = particle_ptr->getData()->map.prediction_reigon.max_point.x;
    p.y = particle_ptr->getData()->map.prediction_reigon.min_point.y;
    p.z = 0;
    line_strip.points.push_back(p);
  }

  {
    geometry_msgs::Point p;
    p.x = particle_ptr->getData()->map.prediction_reigon.min_point.x;
    p.y = particle_ptr->getData()->map.prediction_reigon.min_point.y;
    p.z = 0;
    line_strip.points.push_back(p);
  }

  {
    geometry_msgs::Point p;
    p.x = particle_ptr->getData()->map.prediction_reigon.min_point.x;
    p.y = particle_ptr->getData()->map.prediction_reigon.max_point.y;
    p.z = 0;
    line_strip.points.push_back(p);
  }

  {
    geometry_msgs::Point p;
    p.x = particle_ptr->getData()->map.prediction_reigon.max_point.x;
    p.y = particle_ptr->getData()->map.prediction_reigon.max_point.y;
    p.z = 0;
    line_strip.points.push_back(p);
  }

  particle_border_pub_.publish(line_strip);
}

pcl::PointCloud<pcl::PointXYZI>::Ptr ParticlePublisher::assembleCloud(ParticlePtr_t particle_ptr){
  pcl::PointCloud<pcl::PointXYZI>::Ptr out;
  if(particle_ptr->isRoot()){
    out.reset(new pcl::PointCloud<pcl::PointXYZI>);
    *out = particle_ptr->getData()->map.projected_cloud;
  }else {
    out = assembleCloud(particle_ptr->getParent());
    *out += particle_ptr->getData()->map.projected_cloud;
  }
  return  out;

}


geometry_msgs::PoseArray::Ptr ParticlePublisher::assembleOdomHypothesis(ParticlePtr_t particle_ptr){
  geometry_msgs::PoseArray::Ptr out;
  if(particle_ptr->isRoot()){
    out.reset(new geometry_msgs::PoseArray);
    out->header.frame_id=particle_ptr->getData()->getFinalHypothesis()->header.frame_id;
    out->header.stamp = ros::Time::now();
  }else {
    out = assembleOdomHypothesis(particle_ptr->getParent());
  }
  for(auto odom: particle_ptr->getData()->nav.hypothesis){
    out->poses.push_back(odom->pose.pose);
  }
  return  out;
}

geometry_msgs::PoseArray::Ptr ParticlePublisher::assembleAncestorOdom(ParticlePtr_t particle_ptr, geometry_msgs::PoseArray::Ptr out){
  if(!out){
    out.reset(new geometry_msgs::PoseArray);
    out->header.frame_id=particle_ptr->getData()->getFinalHypothesis()->header.frame_id;
    out->header.stamp = ros::Time::now();
  }
  for(auto odom: particle_ptr->getData()->nav.hypothesis){
    out->poses.push_back(odom->pose.pose);
  }
  for(auto child_ptr: particle_ptr->getChildren()){
    assembleAncestorOdom(child_ptr,out);
  }
  return out;
}




}}
