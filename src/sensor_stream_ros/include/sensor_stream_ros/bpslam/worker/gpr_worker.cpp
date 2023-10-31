#include "gpr_worker.h"
namespace ss{ namespace bpslam {

GPRWorker::GPRWorker(ParticlePtr_t particle, gpr::GprParams params, PredictionReigon::Ptr pred_reigon)
{
  if(pred_reigon==nullptr){
    pred_reigon_.reset(new PredictionReigon);
    *pred_reigon_ = particle->getData()->map.prediction_reigon;
  }else{
    pred_reigon_=pred_reigon;
  }

  particle_ = particle;
  gpr_params_= params;
  output.did_run = false;

  node_.reset(new ros::NodeHandle("~"));
  im_trans_.reset(new image_transport::ImageTransport(*node_));
  //sparsity_pub_ = im_trans_->advertise("viz/sparsity",1);
}

void GPRWorker::setupGPR(){
  particle_->getData()->map.likelihood_by_point.reset();
  particle_->getData()->map.predicted_input.reset();
  particle_->getData()->map.predicted_mu.reset();
  particle_->getData()->map.predicted_var.reset();

  particle_->getData()->map.gpr.reset(new BlockGpr);
  particle_->getData()->map.gpr->params=gpr_params_;

  computeAverageTrainingPoint(particle_->getParent());

  subsample_block_size_ = 0;
}



void GPRWorker::run(){
  //if(particle_->getData()->map.likelihood_by_point.rows()==0){  // only run if there's no solution

    setupGPR();


    if(num_training_blocks_>10){  // make sure we have some training points
      //computeMeanFunction(particle_->getParent(),100 ,true);

      PerfTimer add_blocks;
      addBlocks2GPR(particle_->getParent(), true);
      std::cout << "add_blocks: " << add_blocks.elapsed() << std::endl;


      PerfTimer points_likelihood;
      subsampleParticleLikelihood();
      //computePointsLikelihood();
      std::cout << "points_likelihood: " << points_likelihood.elapsed() << std::endl;

      PerfTimer particle_likelihood;
      computeParticleLikelihood();
      std::cout << "particle_likelihood: " << particle_likelihood.elapsed() << std::endl;

      particle_->getData()->map.gpr.reset(); // delete the GPR now that we're done with it to save memory

      output.did_run = true;
    }else {
      output.did_run = false;
    }
  //}
}

void GPRWorker::subsampleParticleLikelihood(){
  particle_->getData()->map.likelihood_by_point.reset();
  particle_->getData()->map.predicted_mu.reset();

  float ratio = .1f;

  pcl::PointCloud<pcl::PointXYZI>::Ptr subsampled_cloud(new pcl::PointCloud<pcl::PointXYZI>);
  *subsampled_cloud = particle_->getData()->map.projected_cloud;


  pcl::RandomSample<pcl::PointXYZI> filter;
  filter.setSample(subsampled_cloud->size()*ratio);
  filter.setInputCloud(subsampled_cloud);
  filter.filter(*subsampled_cloud);

  BlockParams::Ptr params(new BlockParams);
  params->size = subsampled_cloud->size();

  SingleFrameBlock::Ptr subsampled_block(new SingleFrameBlock(params,subsampled_cloud) );

  computeBlockLikelihood(subsampled_block);
}

void GPRWorker::computePointsLikelihood(){
  particle_->getData()->map.likelihood_by_point.reset();
  particle_->getData()->map.predicted_mu.reset();

  for(auto block : particle_->getData()->map.projected_blocks){
    computeBlockLikelihood(block);
  }

}

void GPRWorker::computeBlockLikelihood(SingleFrameBlock::Ptr block){
  TrainingBlock observation = block->getTrainingData();

  PerfTimer gpr_pred;
  gpr::Prediction estimate = particle_->getData()->map.gpr->predict(observation.x);
  //std::cout << "  gpr_pred: " << gpr_pred.elapsed() << std::endl;

  PerfTimer prepare_for_likelihood;
  add(estimate.mu,getAverageTrainingPoint().z);
  observation.variance.reset(observation.y.rows(),observation.y.cols());
  observation.variance.initHost();
  for (size_t i=0;i<observation.variance.rows();i++) {
    observation.variance(i,0) = gpr_params_.regression->sensor_var;  /// \todo estimate varience in a better way
  }
  observation.variance.host2dev();
  //std::cout << "  prepare_for_likelihood: " << prepare_for_likelihood.elapsed() << std::endl;

  PerfTimer estimate_likelihood;
  CudaMat<float> likelihood = estimateLikelihood(estimate.mu,observation.y,estimate.sigma,observation.variance);
  //std::cout << "  estimate_likelihood: " << estimate_likelihood.elapsed() << std::endl;

  //PerfTimer data2host;
  likelihood.dev2host();
  particle_->getData()->map.likelihood_by_point.setBlockParam(likelihood.getDenseParam());
  particle_->getData()->map.likelihood_by_point.pushBackHost(likelihood,particle_->getData()->map.likelihood_by_point.rows(),0);

  estimate.mu.dev2host();
  particle_->getData()->map.predicted_mu.setBlockParam(likelihood.getDenseParam());
  particle_->getData()->map.predicted_mu.pushBackHost(estimate.mu,particle_->getData()->map.predicted_mu.rows(),0);

  estimate.sigma.dev2host();
  particle_->getData()->map.predicted_var.setBlockParam(likelihood.getDenseParam());
  particle_->getData()->map.predicted_var.pushBackHost(estimate.sigma,particle_->getData()->map.predicted_var.rows(),0);

  estimate.points.dev2host();
  //estimate.points.printHost();
  particle_->getData()->map.predicted_input.setBlockParam(estimate.points.getDenseParam());
  particle_->getData()->map.predicted_input.pushBackHost(estimate.points,particle_->getData()->map.predicted_input.rows(),0);
}

void GPRWorker::computeParticleLikelihood(){
  float sum = 0;
  size_t num = 0;
  for(size_t i = 0;
      i < particle_->getData()->map.likelihood_by_point.rows()
          *particle_->getData()->map.likelihood_by_point.getBlockParam().rows;
      i++){
    sum += particle_->getData()->map.likelihood_by_point.getVal(i,0);
    num++;
  }
  particle_->getData()->map.likelihood = sum/num;
  if(std::isnan(particle_->getData()->map.likelihood))
    particle_->getData()->map.likelihood=0;
}

void GPRWorker::computeMeanFunction(ParticlePtr_t other_particle, size_t block_size, bool recursive){
  float margin = 10.0f;
  if(pred_reigon_->doesOverlap(other_particle->getData()->map.prediction_reigon,margin)){
    //std::cout << "adding particle " <<  other_particle->getId() << std::endl;
    for(auto block : other_particle->getData()->map.projected_blocks){
      if(pred_reigon_->doesOverlap(block,margin)){
        //std::cout << "adding block: " <<  block->getCenterOfMass().x << block->getCenterOfMass().y << std::endl;
        //particle_->getData()->map.training_blocks.push_back(block);
        ss::TrainingBlock trainingData = block->getSubsampledTrainingData(block_size);
        add(trainingData.y,-getAverageTrainingPoint().z);
        particle_->getData()->map.mean_function->addTrainingData(trainingData.x,trainingData.y);
      }
    }
  }
  if(recursive){
    if(!other_particle->isRoot()){
      addBlocks2GPR(other_particle->getParent(),recursive);
    }
  }
}

void GPRWorker::addBlocks2GPR(ParticlePtr_t other_particle, bool recursive){
  float margin =  gpr_params_.regression->kernel->hyperparam("length_scale");
  if(pred_reigon_->doesOverlap(other_particle->getData()->map.prediction_reigon,margin)){
    for(auto block : other_particle->getData()->map.projected_blocks){
      if(pred_reigon_->doesOverlap(block,margin)){
        particle_->getData()->map.training_blocks.push_back(block);

        ss::TrainingBlock trainingData;
        if(subsample_block_size_==0)
          trainingData = block->getTrainingData();
        else
          trainingData = block->getSubsampledTrainingData(subsample_block_size_);

        add(trainingData.y,-getAverageTrainingPoint().z);
        particle_->getData()->map.gpr->addTrainingData(trainingData.x,trainingData.y);
        publishSparsity();
      }
    }
  }
  if(recursive){
    if(!other_particle->isRoot()){
      addBlocks2GPR(other_particle->getParent(),recursive);
    }
  }
}

void GPRWorker::publishSparsity(){
//  if(sparsity_pub_.getNumSubscribers()>0){
//      cv::Mat sparsity = ss::viz::sparsity2image(particle_->getData()->map.gpr->getCholeskyFactor(),
//                                                 float(particle_->getData()->map.gpr->getCholeskyFactor().getBlockParam().rows)/float(100));
//      sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", sparsity).toImageMsg();
//      sparsity_pub_.publish(msg);
//  }
}

pcl::PointXYZ GPRWorker::getAverageTrainingPoint(){
  pcl::PointXYZ avg = sum_training_blocks_;
  avg.x=avg.x/num_training_blocks_;
  avg.y=avg.y/num_training_blocks_;
  avg.z=avg.z/num_training_blocks_;
  return avg;
}

void GPRWorker::computeAverageTrainingPoint(ParticlePtr_t other_particle){
  if(!other_particle){
    return;
  }
  if(other_particle->isRoot()){
    sum_training_blocks_.x = 0;
    sum_training_blocks_.y = 0;
    sum_training_blocks_.z = 0;
    num_training_blocks_   = 0;
  }else {
    computeAverageTrainingPoint(other_particle->getParent());
  }
  if(pred_reigon_->doesOverlap(other_particle->getData()->map.prediction_reigon)){
    for(auto block : other_particle->getData()->map.projected_blocks){
      if(pred_reigon_->doesOverlap(block)){
        sum_training_blocks_.x += block->getCenterOfMass().x;
        sum_training_blocks_.y += block->getCenterOfMass().y;
        sum_training_blocks_.z += block->getCenterOfMass().z;
        num_training_blocks_++;
      }
    }
  }
  return;
}

}}
