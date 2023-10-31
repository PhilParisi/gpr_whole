#include "recursive_gpr_worker.h"
namespace ss{ namespace bpslam {


RecursiveGPRWorker::RecursiveGPRWorker(ParticlePtr_t particle, gpr::GprParams params, PredictionReigon::Ptr pred_reigon, float mean, ros::Time time_cutoff, size_t subsample_block_size):
  GPRWorker (particle, params, pred_reigon)
{
  sum_training_blocks_.z = mean;
  num_training_blocks_ = 1;
  time_cutoff_ = time_cutoff;
  subsample_block_size_ = subsample_block_size;
}

RecursiveGPRWorker::~RecursiveGPRWorker(){
  particle_->getData()->map.gpr.reset();
}

void RecursiveGPRWorker::run(){
  if(particle_->isRoot()){
    particle_->getData()->map.gpr.reset(new BlockGpr);
    particle_->getData()->map.gpr->params=gpr_params_;
    particle_->getData()->map.training_blocks.clear();
  }else{
    particle_->getData()->map.gpr.reset(new BlockGpr(*particle_->getParent()->getData()->map.gpr));
  }

  if(particle_->isLeaf()){
    if(particle_->getData()->map.gpr->getCholeskyFactor().nnz()>0){
      subsampleParticleLikelihood();
      computeParticleLikelihood();
      // fill up metrics
      particle_->getData()->metrics.cholesky_nnz = particle_->getData()->map.gpr->getCholeskyFactor().nnz();
      particle_->getData()->metrics.cholesky_block_size = particle_->getData()->map.gpr->getCholeskyFactor().getBlockParam().rows*
                                                          particle_->getData()->map.gpr->getCholeskyFactor().getBlockParam().cols;
      particle_->getData()->metrics.cholesky_bytes = (particle_->getData()->metrics.cholesky_block_size * particle_->getData()->metrics.cholesky_nnz * sizeof(float))
                                                      / 1e9;
    }else {
      particle_->getData()->map.likelihood = NAN;
    }

  } else {
    if(particle_->getData()->nav.start_time.toSec() < time_cutoff_.toSec()){
      addBlocks2GPR(particle_,false);
    }
    for(auto child : particle_->getChildren()){
      child->getData()->map.training_blocks = particle_->getData()->map.training_blocks;
      RecursiveGPRWorker leafWorker(child,gpr_params_,pred_reigon_,sum_training_blocks_.z/num_training_blocks_, time_cutoff_, subsample_block_size_);
      leafWorker.run();
    }
  }




}



size_t RecursiveGPRWorker::getNumTrainingBlocks(){
  size_t count=0;
  ParticlePtr_t current  = particle_;
  float margin =  gpr_params_.regression->kernel->hyperparam("length_scale");
  while(!current->isLeaf()){
    if(current->getData()->nav.start_time.toSec() < time_cutoff_.toSec()){
      if(pred_reigon_->doesOverlap(current->getData()->map.prediction_reigon,margin)){
        for(auto block : current->getData()->map.projected_blocks){
          if(pred_reigon_->doesOverlap(block,margin)){
            count++;
          }
        }
      }
    }
    current = *current->getChildren().begin();
  }
  return count;
}

}}
