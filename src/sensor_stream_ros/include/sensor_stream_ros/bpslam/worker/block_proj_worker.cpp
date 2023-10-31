#include "block_proj_worker.h"
namespace ss{ namespace bpslam {

BlockProjWorker::BlockProjWorker(ss::bpslam::ParticlePtr_t particle,
                                 multibeam_process::TfProjector::Ptr static_projector,
                                 WorkerQueue::Ptr gpr_queue,
                                 gpr::GprParams params)
{
  particle_=particle;
  static_projector_=static_projector;
  gpr_queue_=gpr_queue;
  gpr_params_=params;
}

void BlockProjWorker::run(){
  if(particle_->isRoot()){
    return;
  }
  auto first = particle_->getData()->map.data_front;
  auto last = particle_->getData()->map.data_back;
  std::string base_link = static_projector_->getRobotModel().getRoot()->name;
  std::string odom_frame = particle_->getData()->getOdomFrame();
  size_t i = 0;
  for(auto it = first ; it != last ; it++) {
    try {
      SensorFrameBlock::Ptr sensor_frame_block = *it;
      SensorFrameBlock::Ptr base_link_block = sensor_frame_block->transform(*static_projector_,base_link);
      SingleFrameBlock::Ptr map_block = base_link_block->transform2StaticFrame(*particle_->tf_buffer,odom_frame);
      particle_->getData()->map.projected_blocks.push_back(map_block);
      particle_->getData()->map.projected_cloud += *map_block->getCloud();
    } catch (tf2::ExtrapolationException) {
      ROS_WARN("block was outside of TF information, ignoring...");
    }
    catch (tf2::LookupException e) {
      ROS_WARN("%s", e.what());
    }
  }
  pcl::getMinMax3D (particle_->getData()->map.projected_cloud,
                    particle_->getData()->map.prediction_reigon.min_point,
                    particle_->getData()->map.prediction_reigon.max_point);

//  GPRWorker::Ptr gpr_worker(new GPRWorker(particle_,gpr_params_));
//  gpr_queue_->pushBack(gpr_worker);
  return;
}

}}
