#include "bpslam.h"
namespace ss{ namespace bpslam {

//thread_local std::default_random_engine EKFWorker::generator;

BPSlam::BPSlam()
{
  Eigen::initParallel();
  // reset pointersparticles_spawned_
  node_.reset( new(ros::NodeHandle) );
  static_projector_.reset( new(multibeam_process::TfProjector));
  gpr_queue_.reset( new WorkerQueue);
  // setup publishers
  pose_pub_=node_->advertise<geometry_msgs::PoseArray>("pose_array",1);
  odom_pub_=node_->advertise<nav_msgs::Odometry>("odom",1);
  transformed_cloud_pub_=node_->advertise<sensor_msgs::PointCloud2>("transformed_cloud",1);

  //initalize vars;
  particles_spawned_ = false;

  config_.gpr_params.regression.reset(new gpr::RegressionParams);
  config_.gpr_params.prediction.reset(new gpr::PredParams);
  config_.gpr_params.regression->kernel.reset(new ss::kernels::SqExpSparse2d);
  config_.gpr_params.regression->kernel->hyperparam("length_scale")  = 4.0f;
  config_.gpr_params.regression->kernel->hyperparam("process_noise") = 0.1f;
  config_.gpr_params.regression->kernel->hyperparam2dev();

  config_.gpr_params.regression->sensor_var = powf(0.2f,2);
  config_.gpr_params.regression->nnzThresh=1e-16f;
  config_.steps_between_cull = 2;
  block_array_.params = config_.block_array_params;

  steps_since_cull_ = 0;
  profiling.cholesky_bytes = 0;
  profiling.leaf_set=0;

  profiling.first_ping_time = -1.0;

  report_.reset(new profiling::Report());
  report_->setDescription("A set of timeseries to assess the performance of the BPSLAM");
  per_step_series_.reset(new profiling::Series("per_step_series"));
  particle_series_.reset(new profiling::YamlMetric("particle_metrics"));
  //profiling::YamlMetric test("test");
  report_->addSeries(per_step_series_);
  report_->addMetric(particle_series_);
  most_likely_particle_.reset();
  profiling.update_most_likely_particle=false;
  //per_step_series_->setName("per_step_series");


}

bool BPSlam::ready2spawn(){
  return filter_time_ > spawn_time_ && data_.odom_.size()>0;// wait till the particle has lived long enough)  // make sure we have odometery to work on
}

bool BPSlam::ready2cull(){
  return steps_since_cull_ >= config_.steps_between_cull;
}

void BPSlam::computeLeafParticles(){
  last_leaf_queue_=getLeafQueue();
  for (auto leaf : getLeafQueue()) {
    leaf->getData()->map.gpr_params = config_.gpr_params;
  }
  computeHypothesis();
  projectBlocks();
  publishParticles();

  steps_since_cull_++;
}

void BPSlam::startProfiling(){
  profiling.cholesky_bytes=0.0;
  profiling.loop_timer.reset();
}

void BPSlam::pauseProfiling(){
  profiling.loop_timer.pause();
}

void BPSlam::resumeProfiling(){
  profiling.loop_timer.resume();
}

void assembleOdom(ParticlePtr_t particle, std::vector<double> & x ,std::vector<double> & y){
  if(!particle->isRoot()){
    assembleOdom(particle->getParent(),x,y);
  }
//  for(auto odom: particle->getData()->nav.hypothesis){
//    series_ptr->pushBack("x",odom->pose.pose.position.x);
//    series_ptr->pushBack("y",odom->pose.pose.position.y);
//  }
  x.push_back(particle->getData()->nav.hypothesis.back()->pose.pose.position.x);
  y.push_back(particle->getData()->nav.hypothesis.back()->pose.pose.position.y);
}

void BPSlam::particleQueue2Metrics(){
  bool reset=true;
    for(auto particle: getLeafQueue()){
      if(!particle->isRoot()&&particle->getData()->map.likelihood>0){
        if(reset){
          //particle_series_->data.reset();
          profiling.leaf_set++;
          particle_series_->data["last_resamping_step"]=profiling.leaf_set-1;
          particle_series_->data["last_step_number"]=per_step_series_->size();
          reset=false;
        }
        std::vector<double>  x,  y;
        assembleOdom(particle,x,y);
        particle_series_->data["particles"][std::to_string(particle->getId())]["x"]=x;
        particle_series_->data["particles"][std::to_string(particle->getId())]["y"]=y;
        particle_series_->data["particles"][std::to_string(particle->getId())]["likelihood"]=particle->getData()->map.likelihood;
        particle_series_->data["particles"][std::to_string(particle->getId())]["weight"]=particle->getData()->map.weight;
        particle_series_->data["particles"][std::to_string(particle->getId())]["resamping_step"]=profiling.leaf_set-1;
        particle_series_->data["particles"][std::to_string(particle->getId())]["step_number"]=per_step_series_->size();
        profiling.update_most_likely_particle = false;
      }
    }
}

void BPSlam::addProfilingMetrics(){

  // particle metrics
  //if(most_likely_particle_  &&  profiling.update_most_likely_particle){

  //}

  // per step metrics
  double last_time = 0;
//  if( profiling.start_time<0){
//    profiling.start_time=filter_time_.toSec();
//  }
  if( per_step_series_->size()>0)
    last_time = per_step_series_->getValue("cumulative_time",per_step_series_->size()-1);
  per_step_series_->pushBack("loop_time", profiling.loop_timer.elapsed());
  per_step_series_->pushBack("cumulative_time", last_time + profiling.loop_timer.elapsed());
  per_step_series_->pushBack("latest_ping_time", profiling.last_ping_time - profiling.first_ping_time);

  per_step_series_->pushBack("num_particles", int(getLeafQueue().size()));
  per_step_series_->pushBack("filter_time", filter_time_.toSec());
  profiling.start_time = per_step_series_->getValue("filter_time",0);
  per_step_series_->pushBack("elapsed_time(minutes)", (filter_time_.toSec()-profiling.start_time)/60.0);
  per_step_series_->pushBack("odom_x", data_.odom_.back()->pose.pose.position.x );
  per_step_series_->pushBack("odom_y", data_.odom_.back()->pose.pose.position.y );

  geometry_msgs::Pose avg_pose;
  double x=0;
  double y=0;
  double z=0;
  geometry_msgs::Point pos_error;
  geometry_msgs::Point pos_var;
  double magnitude_error;
  double magnitude_var = 0;
  double var_x = 0;
  double var_y = 0;
  double var_z = 0;
  for(auto particle_prt : getLeafQueue()){
    geometry_msgs::Pose avg_pose;
    auto latest = particle_prt->getData()->nav.hypothesis.back()->pose.pose.position;
    x += latest.x;
    y += latest.y;
    z += latest.z;
  }
  avg_pose.position.x=x/getLeafQueue().size();
  avg_pose.position.y=y/getLeafQueue().size();
  avg_pose.position.z=z/getLeafQueue().size();

  pos_error.x = data_.odom_.back()->pose.pose.position.x - avg_pose.position.x;
  pos_error.y = data_.odom_.back()->pose.pose.position.y - avg_pose.position.y;
  pos_error.z = data_.odom_.back()->pose.pose.position.z - avg_pose.position.z;

  magnitude_error = sqrt(pow(pos_error.x,2)+
                         pow(pos_error.y,2)+
                         pow(pos_error.z,2));

  for(auto particle_prt : getLeafQueue()){
    geometry_msgs::Point particle_error;
    particle_error.x = particle_prt->getData()->nav.hypothesis.back()->pose.pose.position.x - avg_pose.position.x;
    particle_error.y = particle_prt->getData()->nav.hypothesis.back()->pose.pose.position.y - avg_pose.position.y;
    particle_error.z = particle_prt->getData()->nav.hypothesis.back()->pose.pose.position.z - avg_pose.position.z;

    var_x += pow(particle_error.x,2);
    var_y += pow(particle_error.y,2);
    var_z += pow(particle_error.z,2);
    double particle_magnitude_error = sqrt(pow(particle_error.x,2)+
                                           pow(particle_error.y,2)+
                                           pow(particle_error.z,2));

    magnitude_var += pow(particle_magnitude_error,2);
  }

  var_x = var_x/getLeafQueue().size();
  var_y = var_y/getLeafQueue().size();
  var_z = var_z/getLeafQueue().size();

  magnitude_var = magnitude_var/getLeafQueue().size();

//  magnitude_var = sqrt(pow(var_x,)
  per_step_series_->pushBack("magnitude_error", magnitude_error,magnitude_var);
  per_step_series_->pushBack("max_cholesky_size", profiling.cholesky_bytes);
  per_step_series_->pushBack("avg_particle_x", avg_pose.position.x, var_x);
  per_step_series_->pushBack("avg_particle_y", avg_pose.position.y, var_y);
  if(config_.metrics_file!=""){
    saveMetrics(config_.metrics_file);
  }
}

void BPSlam::saveMetrics(std::string fName){
  YAML::Node series_yaml = report_->toYaml();
  boost::filesystem::path filename(fName);
  if(!boost::filesystem::exists(filename.parent_path())){
    boost::filesystem::create_directory(filename.parent_path());
  }

  std::ofstream fout(fName);
  fout << series_yaml;

}


void BPSlam::spinOnce(){

  if(ready2spawn())
  {
    computeLeafParticles();
    spawnParticles();
    particles_spawned_=true;
  }else {
    particles_spawned_=false;
  }
}


void BPSlam::addSensorPing(const PointCloudPtr &ping){
  assert(ping!=nullptr);
  if(profiling.first_ping_time<=0){
    profiling.first_ping_time=ping->header.stamp.toSec();
  }
  profiling.last_ping_time=ping->header.stamp.toSec();
  block_array_.addPing(ping);
  return;
}

void BPSlam::addOdometry(const nav_msgs::Odometry::ConstPtr &odom_msg){
  data_.odom_.push_back(odom_msg);
  filter_time_ = odom_msg->header.stamp;
  if(data_.odom_.size()==1){ // if this is our first message
    rootParticlePtr()->getData()->nav.odom_front=data_.odom_.begin();
    rootParticlePtr()->getData()->nav.start_time=filter_time_;
    nav_msgs::Odometry::Ptr initial_hyp(new nav_msgs::Odometry);
    *initial_hyp=*odom_msg;
    rootParticlePtr()->getData()->nav.hypothesis.push_back(initial_hyp);
    spawn_time_=filter_time_;
  }

  return;
}

void BPSlam::computeHypothesis(){
  assert(filter_time_ > spawn_time_ && data_.odom_.size()>0);
  //std::cout<<"Computing hypothesis...";
  workerQueue_.run(); // start queue running
  for (auto leaf : getLeafQueue()) {
    leaf->getData()->nav.end_time=filter_time_;
    leaf->getData()->nav.odom_back=std::prev(data_.odom_.end());
    EKFWorker::Ptr worker(new EKFWorker(leaf,config_.ekf_params));
    //worker->run();
    workerQueue_.pushBack(worker);
  }
  workerQueue_.addSyncCommand();
  workerQueue_.sync();


}

void BPSlam::computeLeafGpr(){
  PredictionReigon::Ptr reigon(new PredictionReigon);
  auto first_leaf=*getLeafQueue().begin();
  reigon->max_point = first_leaf->getData()->map.prediction_reigon.max_point;
  reigon->min_point = first_leaf->getData()->map.prediction_reigon.min_point;
  auto leaf_queue = getLeafQueue();

  for(auto leaf : leaf_queue){
    if(reigon->max_point.x < leaf->getData()->map.prediction_reigon.max_point.x)
      reigon->max_point.x = leaf->getData()->map.prediction_reigon.max_point.x;
    if(reigon->max_point.y < leaf->getData()->map.prediction_reigon.max_point.y)
      reigon->max_point.y = leaf->getData()->map.prediction_reigon.max_point.y;
    if(reigon->max_point.z < leaf->getData()->map.prediction_reigon.max_point.z)
      reigon->max_point.z = leaf->getData()->map.prediction_reigon.max_point.z;

    if(reigon->min_point.x > leaf->getData()->map.prediction_reigon.max_point.x)
      reigon->min_point.x = leaf->getData()->map.prediction_reigon.max_point.x;
    if(reigon->min_point.y > leaf->getData()->map.prediction_reigon.max_point.y)
      reigon->min_point.y = leaf->getData()->map.prediction_reigon.max_point.y;
    if(reigon->min_point.z > leaf->getData()->map.prediction_reigon.max_point.z)
      reigon->min_point.z = leaf->getData()->map.prediction_reigon.max_point.z;
  }
  ros::Time cutoff_time = getLeafQueue().begin()->operator->()->getData()->nav.start_time - config_.min_model_particle_age;

  try{
    bool ready2run = true;
    RecursiveGPRWorker root_worker(rootParticlePtr(),config_.gpr_params,reigon,0,cutoff_time);
    size_t training_blocks = root_worker.getNumTrainingBlocks();
    size_t num_points = training_blocks * block_array_.params.block_params->size;
    if(num_points>config_.max_training_points){
      //double ratio = double(config_.max_training_points)/double(num_points);
      size_t block_size = config_.max_training_points * block_array_.params.block_params->size / num_points;
      block_size = block_size/4;
      block_size = block_size*4;
      ROS_WARN("The number of training points (%zu) exceeds the max_training_points parameter (%zu) approximating with reduced block size: %zu",num_points,config_.max_training_points, block_size);
      if(block_size > config_.min_block_size){
        root_worker.setSubsampleTrainingBlockSize(block_size);
      }else{
        ROS_WARN("The block size (%zu) is less than the  min_block_size parameter (%zu). Ignoring",block_size,config_.min_block_size);
        ready2run = false;
      }
    }
    if(ready2run){
      root_worker.computeAverageTrainingPoint(first_leaf->getParent());
      root_worker.run();
      config_.gpr_params.regression->kernel->hyperparam2host();
    }
  }
  catch (...) {
    ROS_ERROR("Error in GPR worker: %s",boost::current_exception_diagnostic_information().c_str());
  }
  for(auto particle_ptr : getLeafQueue()){
    profiling.cholesky_bytes = std::max(profiling.cholesky_bytes, particle_ptr->getData()->metrics.cholesky_bytes);
  }
}

void BPSlam::projectBlocks(){
  assert(filter_time_ > spawn_time_);
  workerQueue_.run(-1); // run nSystemThreads-1 threads
  //gpr_queue_->run(1);   // run one thread for the GPR queue
  for (auto leaf : getLeafQueue()) {
    if(leaf->isRoot()){  // if we are the root particle
      if(block_array_.size()==0){
        block_array_.addEmptyBlock();
      }
      leaf->getData()->map.data_back=std::prev(block_array_.end());
      leaf->getData()->map.data_front=block_array_.begin();
    }else{
      leaf->getData()->map.data_back=std::prev(block_array_.end());
      leaf->getData()->map.data_front=leaf->getParent()->getData()->map.data_back;
    }
    BlockProjWorker::Ptr worker(new BlockProjWorker(leaf,
                                                    static_projector_,
                                                    gpr_queue_,
                                                    config_.gpr_params));
    //worker->run();
    workerQueue_.pushBack(worker);
  }

  workerQueue_.addSyncCommand();
  workerQueue_.sync();
  //gpr_queue_->addSyncCommand();

  return;
}

void BPSlam::publishParticles(){

  geometry_msgs::PoseArray pose_array;
  pose_array.header.stamp=ros::Time::now();
  pose_array.header.frame_id = data_.odom_.front()->header.frame_id;
  for (auto leaf : getLeafQueue()) {
    pose_array.poses.push_back(geometry_msgs::Pose());
    pose_array.poses.back()=leaf->getData()->getFinalHypothesis()->pose.pose;
  }
  nav_msgs::Odometry odom_msg = *data_.odom_.back();
  odom_msg.header.stamp = pose_array.header.stamp;
  odom_pub_.publish(odom_msg);
  pose_pub_.publish(pose_array);
}

void BPSlam::spawnParticles(){

  size_t queue_size = getLeafQueue().size();
  size_t num2spawn = 0;
  if(ready2cull()){
    cullParticles();
    num2spawn = config_.particle.n_children;
  }else{
    num2spawn = 1;
  }

  for (auto particle_ptr : getLeafQueue()){
    if(queue_size>=config_.max_particles){
      num2spawn=1;
    }
    addLeafs(particle_ptr,num2spawn);
    removeLeaf(particle_ptr);
  }
  spawn_time_=filter_time_ + config_.particle.lifespan2Duration();

}


struct SortContainer{
  SortContainer(ParticlePtr_t ptr){particle_ptr=ptr;}
  ParticlePtr_t particle_ptr;
  bool operator <(const SortContainer & other) const {return particle_ptr->getData()->map.weight < other.particle_ptr->getData()->map.weight;}
};

void BPSlam::cullParticles(size_t remaining){

  //
  // update the particles
  //
  computeLeafGpr();

  //
  // Itterate through all particles and assign weights
  //
  float sum_likelihood=0;
  float num_particles =0;
  float min_likelihood=INFINITY;
  float max_likelihood=0;
  for(auto particle_ptr: getLeafQueue()){
    if(particle_ptr->getData()->numTrainingBlocks()>0){
      sum_likelihood += particle_ptr->getData()->map.likelihood;
      num_particles ++;
      max_likelihood = std::max(particle_ptr->getData()->map.likelihood,max_likelihood);
      min_likelihood = std::min(particle_ptr->getData()->map.likelihood,min_likelihood);
    }
  }
  // assign weights by normalizing the likelihood
  for(auto particle_ptr: getLeafQueue()){
    particle_ptr->getData()->map.weight = (particle_ptr->getData()->map.likelihood - min_likelihood)/(max_likelihood-min_likelihood);
  }

  //
  // sort the particles by weight
  //
  std::set<SortContainer> sorted_leaf_queue;
  for(auto particle_ptr: getLeafQueue()){
    sorted_leaf_queue.insert(particle_ptr);
  }
  most_likely_particle_ = sorted_leaf_queue.rbegin()->particle_ptr;
  profiling.update_most_likely_particle=true;


  //
  // add queue to metrics before kill
  //
  particleQueue2Metrics();
  //
  // kill particles
  //
  for(auto particle_container: sorted_leaf_queue){
    if(getLeafQueue().size()<config_.min_particles)
      break;

    auto particle_ptr = particle_container.particle_ptr;
    srand( unsigned(particle_ptr->getId()) );
    double random = (double(rand()) / (RAND_MAX));
    if(particle_ptr->getData()->map.weight < random && particle_ptr->getData()->map.likelihood > 0){
      removeAncestors(particle_ptr);
      removeLeaf(particle_ptr);
    }
  }

  steps_since_cull_ = 0;







/*
  double sum = 0;
  double sum_of_squares = 0;
  size_t num = 0;
  double average = 0;
  double stdev = 0;
  computeLeafGpr();
  for(auto particle_ptr: getLeafQueue()){
    if(particle_ptr->getData()->numTrainingBlocks()>0){
      sum += particle_ptr->getData()->map.likelihood;
      sum_of_squares += pow(particle_ptr->getData()->map.likelihood,2);
      num ++;
    }
  }

  if(num==0){
    return; // no vaild traing data found;
  }

  average = sum/num;
  stdev = sqrt(sum_of_squares/num);
  for(auto particle_ptr: getLeafQueue()){
    if(particle_ptr->getData()->map.likelihood<average){
      //particle_ptr->removeFromTree();
      removeAncestors(particle_ptr);
      removeLeaf(particle_ptr);
    }
  }

  steps_since_cull_ = 0;
  */
}



void BPSlam::addLeafs(ParticlePtr_t parent, size_t n){
  if(n==0)
    n=config_.particle.n_children;
  for (size_t i=0 ; i<n; i++) {
    ParticlePtr_t new_particle;
    new_particle.reset(new Particle_t);
    new_particle->getData()->nav.start_time=filter_time_;
    new_particle->getData()->nav.odom_front=parent->getData()->nav.odom_back;
    addLeaf(new_particle,parent);
  }

}

void BPSlam::optimizeGpr(ParticlePtr_t particle){
  gpr::HPOptimizer optimizer;
  GPRWorker worker(particle,config_.gpr_params);
  worker.setupGPR();
  worker.addBlocks2GPR(particle->getParent());
  optimizer.optimize(config_.gpr_params,particle);
}

}}
