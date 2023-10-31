#ifndef BPSLAM_PARTICLE_DATA_H
#define BPSLAM_PARTICLE_DATA_H
#include "pf_base.hpp"
#include "../common/blockarray.h"
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include "../common/sensorframeblock.h"
#include "../common/single_frame_block.h"
#include <sensor_stream/include/sensor_stream/blockgpr.h>

namespace ss{ namespace bpslam {



typedef  sensor_msgs::PointCloud2::ConstPtr PointCloudPtr;

class PredictionReigon{
public:
  typedef  std::shared_ptr<PredictionReigon> Ptr;
  bool doesOverlap(PredictionReigon & other, float margin = 0); //!< \brief do the projected clouds
  bool doesOverlap(SingleFrameBlock::Ptr block, float margin = 0);
  pcl::PointXYZI min_point;
  pcl::PointXYZI max_point;
};

/*!
 * \brief The MapData struct contains info pertaining to multibeam and map creation
 */
struct MapData{
  std::list<SensorFrameBlock::Ptr>::iterator data_front;
  std::list<SensorFrameBlock::Ptr>::iterator data_back;
  std::shared_ptr<BlockGpr> gpr;
  std::shared_ptr<BlockGpr> mean_function;
  pcl::PointCloud<pcl::PointXYZI> projected_cloud;
  std::deque<SingleFrameBlock::Ptr> projected_blocks;
//  pcl::PointXYZI min_point;
//  pcl::PointXYZI max_point;
  PredictionReigon prediction_reigon;
  gpr::GprParams gpr_params;
  float likelihood;
  float weight;
  CudaBlockMat<float> likelihood_by_point;
  CudaBlockMat<float> predicted_input;
  CudaBlockMat<float> predicted_mu;
  CudaBlockMat<float> predicted_var;
  std::deque<SingleFrameBlock::Ptr> training_blocks;
};

/*!
 * \brief The NavData struct contains data pertaining to the EKF and navigation solution
 */
struct NavData{
  ros::Time start_time;
  ros::Time end_time;
  std::list<nav_msgs::Odometry::ConstPtr>::iterator odom_front;
  std::list<nav_msgs::Odometry::ConstPtr>::iterator odom_back;
  std::deque<nav_msgs::Odometry::Ptr> hypothesis;
};



class BPSLAMParticleData : public ParticleData{
public:
  typedef std::shared_ptr<BPSLAMParticleData> Ptr;
  BPSLAMParticleData(){
    type="BPSLAMParticleData";
    version=1;
    map.likelihood=0;
    metrics.cholesky_nnz=0;
    metrics.cholesky_block_size=0;
    metrics.cholesky_bytes=0;
    map.likelihood = -1.0f;
  }
  MapData map;
  NavData nav;
  struct{
    size_t cholesky_nnz;
    size_t cholesky_block_size;
    double cholesky_bytes;
  } metrics;
  nav_msgs::Odometry::ConstPtr getFinalHypothesis(){return nav.hypothesis.back();}
  std::string getOdomFrame(){return nav.odom_front->get()->header.frame_id;}
  bool doesOverlap(BPSLAMParticleData::Ptr other, float margin = 0); //!< \brief do the projected clouds
  bool doesOverlap(SingleFrameBlock::Ptr block, float margin = 0);
  size_t getBlockSize(){return map.data_front->operator->()->getParams()->size;}
  size_t numTrainingBlocks(){return map.training_blocks.size();}
};

struct BPSLAMFilterData{
  std::deque<PointCloudPtr> sensor_;
  std::list<nav_msgs::Odometry::ConstPtr> odom_;
};
typedef Particle<BPSLAMParticleData>::Ptr ParticlePtr_t;
typedef Particle<BPSLAMParticleData> Particle_t;
}}


#endif // BPSLAM_PARTICLE_DATA_H
