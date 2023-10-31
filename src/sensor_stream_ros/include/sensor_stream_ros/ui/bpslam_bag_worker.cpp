#include "bpslam_bag_worker.h"

BPSLAMBagWorker::BPSLAMBagWorker(ss::bpslam::BagProcessor::Ptr bag_processor_ptr)
{
  bag_processor_ptr_ = bag_processor_ptr;
  timer_ = new QTimer(this);
  timer_->setInterval(0);
  timer_->stop();
  connect(timer_, SIGNAL(timeout()), this, SLOT(iterate()));
  run_mode_ = paused;
}

void BPSLAMBagWorker::start(){
  run_mode_ = step_continuous;
  timer_->start();
}

void BPSLAMBagWorker::onGuiUpdate(){
  if(run_mode_ == step_continuous)
    timer_->start();
}

void BPSLAMBagWorker::stepOnce(){
  run_mode_ = step_once;
  do{
    iterate();
  }while(!bag_processor_ptr_->pf_->ready2spawn());
  bag_processor_ptr_->pf_->pauseProfiling();
}

void BPSLAMBagWorker::cullParticles(){
  if(bag_processor_ptr_->pf_->ready2spawn()){
    bag_processor_ptr_->pf_->cullParticles();
    ready2spawn();
  }
}

void BPSLAMBagWorker::computeLeafsGPR(){
  bag_processor_ptr_->pf_->computeLeafGpr();
//  for(auto leaf_particle : bag_processor_ptr_->pf_->getLeafQueue()){
//    ss::bpslam::GPRWorker worker(leaf_particle, leaf_particle->getData()->map.gpr_params);
//    worker.run();
//  }
}

void BPSLAMBagWorker::pause(){
  run_mode_ = paused;
  timer_->stop();
  bag_processor_ptr_->pf_->pauseProfiling();
}

void BPSLAMBagWorker::iterate(){
//  try {
    bag_processor_ptr_->pf_->resumeProfiling();
    if(bag_processor_ptr_->pf_->ready2spawn()){
      bag_processor_ptr_->pf_->startProfiling();
      bag_processor_ptr_->pf_->spawnParticles();
    }
    if(bag_processor_ptr_->readNext()){
      if(bag_processor_ptr_->pf_->ready2spawn()){
        bag_processor_ptr_->pf_->computeLeafParticles();
        bag_processor_ptr_->pf_->addProfilingMetrics();
        //if(run_mode_==step_continuous)
        timer_->stop();
        emit(ready2spawn());
      }
    }else{
      pause();
    }

//  } catch (...) {
//    pause();
//    QMessageBox msgBox;
//    msgBox.setText(
//          QString::fromStdString(
//            boost::current_exception_diagnostic_information()
//          )
//    );
//    msgBox.exec();
//  }
}
