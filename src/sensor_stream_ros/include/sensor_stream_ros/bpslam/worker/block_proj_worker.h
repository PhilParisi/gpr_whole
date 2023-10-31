#ifndef BLOCKPROJWORKER_H
#define BLOCKPROJWORKER_H

#include "particle_worker.h"
#include "../bpslam_particle_data.h"
#include <multibeam_process_core/tf_projector.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/common/common.h>
#include "gpr_worker.h"

namespace ss{ namespace bpslam {
/*!
 * \brief This ParticleWorker transforms the particle sensor data to the odom frame and adds the results to the gpr_queue
 */
class BlockProjWorker : public ParticleWorker
{
public:
  BlockProjWorker(ParticlePtr_t particle, multibeam_process::TfProjector::Ptr static_projector, WorkerQueue::Ptr gpr_queue, gpr::GprParams params);
  void run();
protected:
  ParticlePtr_t particle_;
  multibeam_process::TfProjector::Ptr static_projector_;
  WorkerQueue::Ptr gpr_queue_;
  gpr::GprParams gpr_params_;
};
}}
#endif // BLOCKPROJWORKER_H
