#ifndef RECURSIVE_GPR_WORKER_H
#define RECURSIVE_GPR_WORKER_H

#include "gpr_worker.h"
#include "../../ss_cv.h"

namespace ss{ namespace bpslam {

class RecursiveGPRWorker : public GPRWorker
{
public:
  RecursiveGPRWorker(ParticlePtr_t particle, gpr::GprParams params, PredictionReigon::Ptr pred_reigon, float mean, ros::Time time_cutoff, size_t subsample_block_size = 0);
  ~RecursiveGPRWorker();
  void run();
  size_t getNumTrainingBlocks();
protected:
  //float mean_;
  ros::Time time_cutoff_;
};

}}

#endif // RECURSIVE_GPR_WORKER_H
