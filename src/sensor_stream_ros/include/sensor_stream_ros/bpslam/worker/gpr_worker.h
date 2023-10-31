#ifndef GPR_WORKER_H
#define GPR_WORKER_H

#include "particle_worker.h"
#include "../bpslam_particle_data.h"
#include <sensor_stream/include/sensor_stream/gpr/gpr_tools.h>
#include <sensor_stream/include/sensor_stream/perf_timer.h>
#include <pcl/filters/random_sample.h>
#include <sensor_stream/include/sensor_stream/gpr/sqexpsparse2d.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_stream_ros/ss_cv.h>

//#include "gpr_worker_cuda.h"


namespace ss{ namespace bpslam {
class GPRWorker : public ParticleWorker
{
public:
  GPRWorker(ParticlePtr_t particle, gpr::GprParams params, PredictionReigon::Ptr pred_reigon =nullptr);
  void setupGPR();
  void computeMeanFunction(ParticlePtr_t other_particle, size_t block_size, bool recursive = false);
  void addBlocks2GPR(ParticlePtr_t other_particle, bool recursive = false);
  virtual void run();
  struct{
    bool did_run;
    size_t overlapping_particles;
  } output;
  void computeAverageTrainingPoint(ParticlePtr_t other_particle);
  void publishSparsity();
  void setSubsampleTrainingBlockSize(size_t size){subsample_block_size_=size;}
protected:
  pcl::PointXYZ getAverageTrainingPoint();

  void computePointsLikelihood();
  void subsampleParticleLikelihood();
  void computeBlockLikelihood(SingleFrameBlock::Ptr block);
  void computeParticleLikelihood();

  ParticlePtr_t particle_;
  gpr::GprParams gpr_params_;
  pcl::PointXYZ sum_training_blocks_;
  size_t num_training_blocks_;
  size_t particle_depth_;
  PredictionReigon::Ptr pred_reigon_;

  ros::NodeHandlePtr node_;
  std::shared_ptr<image_transport::ImageTransport> im_trans_;

  size_t subsample_block_size_;
  //image_transport::Publisher sparsity_pub_;
};
}}
#endif // GPR_WORKER_H
