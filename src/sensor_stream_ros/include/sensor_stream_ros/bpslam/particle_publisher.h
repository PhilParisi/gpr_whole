#ifndef PARTICLE_PUBLISHER_H
#define PARTICLE_PUBLISHER_H

#include "bpslam.h"
#include <pcl/common/common.h>
#include <visualization_msgs/Marker.h>
namespace ss{ namespace bpslam {

class ParticlePublisher
{
public:
  ParticlePublisher();
  void publishProjected(ParticlePtr_t particle_ptr);
  void publishOdomHypothesis(ParticlePtr_t particle_ptr);
  void publishAncestorOdom(ParticlePtr_t particle_ptr);
  void publishGprPrediction(ParticlePtr_t particle_ptr, bool recompute = false);
  void publishMeanFunction(ParticlePtr_t particle_ptr, bool recompute = false);
  void PublishGPRTraining(ParticlePtr_t particle_ptr);
  void publishTrainingBoundary(ParticlePtr_t particle_ptr);
  void saveParticleMap(ParticlePtr_t particle_ptr, std::string filename);
private:
  pcl::PointCloud<pcl::PointXYZI>::Ptr assembleCloud(ParticlePtr_t particle_ptr);
  geometry_msgs::PoseArray::Ptr assembleOdomHypothesis(ParticlePtr_t particle_ptr);
  geometry_msgs::PoseArray::Ptr assembleAncestorOdom(ParticlePtr_t particle_ptr, geometry_msgs::PoseArray::Ptr out=nullptr);
  ros::NodeHandlePtr node_;
  ros::Publisher transformed_cloud_pub_;
  ros::Publisher pose_pub_;
  ros::Publisher gpr_pub_;
  ros::Publisher gpr_training_pub_;
  ros::Publisher particle_border_pub_;
};

}}
#endif // PARTICLE_PUBLISHER_H
