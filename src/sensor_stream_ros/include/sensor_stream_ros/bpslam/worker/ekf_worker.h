#ifndef EKF_WORKER_H
#define EKF_WORKER_H

// local deps
#include "particle_worker.h"
#include "../bpslam_particle_data.h"
#include "../indexing.h"

// STD deps
#include <random>

// ROS deps
#include <geometry_msgs/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/convert.h>

namespace ss{ namespace bpslam {

struct EKFParams{
  std::vector<idx::covIndex> random_vars;
  double uncertainty_multiplier;
  typedef std::shared_ptr<EKFParams> Ptr;
  typedef std::shared_ptr<EKFParams const > ConstPtr;
  EKFParams(){
    uncertainty_multiplier = 1.0;
  }
};

class EKFWorker : public ParticleWorker{
public:
  EKFWorker();
  EKFWorker(ParticlePtr_t particle, EKFParams::ConstPtr params);
  virtual void run();
  void sample();
  void integrate();
  nav_msgs::Odometry::Ptr integrate(nav_msgs::Odometry::ConstPtr last, nav_msgs::Odometry::ConstPtr reference);

  struct{
    ParticlePtr_t particle;
    EKFParams::ConstPtr params;
  }input;

  struct{
    double velocity_error[6];
  }output;
};

}}

#endif // EKF_WORKER_H
