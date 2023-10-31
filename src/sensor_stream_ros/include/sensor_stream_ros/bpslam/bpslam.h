#ifndef BPSLAM_H
#define BPSLAM_H

#include "pf_base.hpp"
#include <ros/time.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseArray.h>

#include <thread>
#include <tf2/LinearMath/Transform.h>

#include <sensor_msgs/PointCloud2.h>
#include "../common/blockarray.h"
#include <pcl/filters/extract_indices.h>
#include <sensor_stream/include/sensor_stream/gpr/sqexpsparse2d.h>
#include "../common/hp_optimizer.h"
#include <algorithm>
#include <boost/exception/diagnostic_information.hpp>


// local
#include "bpslam_particle_data.h"
#include "worker/block_proj_worker.h"
#include "worker/recursive_gpr_worker.h"
#include "worker/ekf_worker.h"
#include "worker/particle_worker.h"
#include <sensor_stream_ros/profiling/report.h>


// other packages
#include <multibeam_process_core/tf_projector.h>

// you need to explicitly include the message types you want to convert
// so the templated convert.h can link properly
#include <geometry_msgs/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/convert.h>


namespace ss{ namespace bpslam {

struct BPSLAMConfig{
  BPSLAMConfig(){
    particle.lifespan=10;
    particle.n_children=2;
    max_particles=64;
    min_particles=8;
    ekf_params.reset(new EKFParams);
    min_model_particle_age.fromSec(20.0); //sec
    block_array_params.block_params.reset(new BlockParams);
    block_array_params.block_params->size = 800;
    max_training_points = 60000;
    min_block_size = 400;
    metrics_file="";
  }
  struct{
    float lifespan;
    size_t n_children;
    ros::Duration lifespan2Duration(){ros::Duration out = ros::Duration(double(lifespan)); return out;}
  } particle;
  size_t max_particles;
  size_t min_particles;
  size_t max_training_points;
  size_t steps_between_cull;
  size_t min_block_size;
  ros::Duration min_model_particle_age;
  EKFParams::Ptr ekf_params;
  gpr::GprParams gpr_params;
  BlockArrayParams block_array_params;
  std::string metrics_file;




};

class BPSlam : public PFBase<BPSLAMParticleData>{
public:
  typedef std::shared_ptr<BPSlam> Ptr;
  BPSlam();

  bool ready2spawn();

  bool ready2cull();

  void computeLeafParticles();

  void startProfiling();

  void pauseProfiling();

  void resumeProfiling();

  void addProfilingMetrics();

  void saveMetrics(std::string filename);

  void setMetricsFile(std::string filename){config_.metrics_file = filename;}

  void spinOnce();

  bool particlesSpawned(){return particles_spawned_;}

  /*!
   * \brief load a URDF to populate the static TF model
   * \param urdf_filename
   */
  void addURDF(std::string urdf_filename){static_projector_->loadRobotModel(urdf_filename);}

  /*!
   * \brief pushes a sensor ping to the bakc of the queue
   * \param ping a ss::bpslam::PointCloudPtr to a single sensor return in sensor frame
   */
  void addSensorPing(const PointCloudPtr& ping);

  /*!
   * \brief callback for an odometry message from the external EKF
   * \param odom_msg odometry from the EKF node
   */
  void addOdometry(const nav_msgs::Odometry::ConstPtr& odom_msg);

  /*!
   * \brief Add leafs to to the specified particle accoring to BPSlam::BPSLAMConfig then remove
   * the particle from the leaf set
   * \param parent
   */
  void addLeafs(ParticlePtr_t parent, size_t n = 0);

  /*!
   * \brief Compute the hypothesis for each particle in the set of children
   */
  void computeHypothesis();
  /*!
   * \brief Recursively computes the GPR solution for all the leaf particles
   */
  void computeLeafGpr();
  void projectBlocks();
  void spawnParticles();
  void spawnParticles(size_t n);
  void cullParticles(size_t remaining = 0);
  void publishParticles();
  void optimizeGpr(ParticlePtr_t particle);
  BPSLAMFilterData getInputData(){return data_;}
  const std::unordered_set<ParticlePtr_t> & getFinishedLeafQueue(){return last_leaf_queue_;}

  BPSLAMConfig config_;
  profiling::Report::Ptr getReportPtr(){return report_;}
protected:
  void particleQueue2Metrics();
  std::unordered_set<ParticlePtr_t> last_leaf_queue_;
  std::shared_ptr<std::mutex> rand_mtx_;
  ros::NodeHandlePtr node_;
  ros::Publisher pose_pub_;
  ros::Publisher odom_pub_;
  ros::Publisher transformed_cloud_pub_;
  WorkerQueue workerQueue_;
  WorkerQueue::Ptr gpr_queue_;
  BPSLAMFilterData data_;
  ros::Time filter_time_; ///< \brief the current time of the filter (based on odom time)
  ros::Time spawn_time_;  ///< \brief the time that a new set of particles should spawn
  //BPSLAMParticleData current_state_;
  ss::BlockArray block_array_;
  multibeam_process::TfProjector::Ptr static_projector_;
  bool particles_spawned_;
  size_t steps_since_cull_;
  profiling::Report::Ptr report_;
  profiling::Series::Ptr per_step_series_;
  profiling::YamlMetric::Ptr particle_series_;
  ParticlePtr_t most_likely_particle_;
  struct{
    double cholesky_bytes;
    PerfTimer loop_timer;
    double first_ping_time;
    double last_ping_time;
    double start_time;
    bool update_most_likely_particle;
    size_t leaf_set;
  } profiling;

};


}}

#endif // BPSLAM_H
