#include "particle_worker.h"
namespace ss{ namespace bpslam {

ParticleWorker::ParticleWorker()
{
  cmd=cmd::run;
}
WorkerQueue::WorkerQueue(){
  running_=false;
}
WorkerQueue::~WorkerQueue(){
  addSyncCommand();
  sync();
}

void WorkerQueue::pushBack(ParticleWorker::Ptr worker){
  std::lock_guard<std::mutex> lock(queue_mutex_);
  worker_queue_.push_back(worker);
}

void WorkerQueue::pushFront(ParticleWorker::Ptr worker){
  std::lock_guard<std::mutex> lock(queue_mutex_);
  worker_queue_.push_front(worker);
}

ParticleWorker::Ptr WorkerQueue::popFront(){
  std::lock_guard<std::mutex> lock(queue_mutex_);
  if(worker_queue_.size()==0){  //if the queue is empty issue a wait command
    WaitCmd::Ptr wait_cmd(new WaitCmd);
    return wait_cmd;
  }
  ParticleWorker::Ptr worker = worker_queue_.front();
  if(worker->cmd == cmd::sync){
    return worker; // this function is not allowed to remove a sync command;
  }else{
    worker_queue_.pop_front();
    return worker;
  }
}

bool WorkerQueue::processFront(){
    ParticleWorker::Ptr worker = popFront();  // we can't do ANYTHING to a worker till we pop it
    switch (worker->cmd) {
    case cmd::wait:
      return true;
    case cmd::sync:
      return false;
    case cmd::run:
      worker->run();
      return true;
    }
}

void WorkerQueue::addSyncCommand(){
  SyncCmd::Ptr sync_cmd(new SyncCmd);
  pushBack(sync_cmd);
}

void WorkerQueue::run(int thread_count){
  if(running_){
    throw std::runtime_error("WorkerQueue::run() was called while the queue was already running!");
  }
  running_=true;
  if(thread_count<=0){
    thread_count = std::thread::hardware_concurrency()+thread_count;
    if(thread_count<=0){
      thread_count=1;
    }
  }
  size_t num_threads = size_t(thread_count);
  for (uint i = 0; i < num_threads; ++i) {
    threads_.push_back(std::thread(&WorkerQueue::runThread,this));
  }
//  runThread();
}

void WorkerQueue::sync(){
  running_=false;
  for (uint i = 0; i < threads_.size(); ++i){
    threads_[i].join();  for (uint i = 0; i < std::thread::hardware_concurrency()-1; ++i) {
    }
  }
  threads_.clear();
  if(worker_queue_.size()>0){
    worker_queue_.pop_front();
  }

//  std::cout<< "\nsync:  ";
//  std::cout<< worker_queue_.size() << std::endl;
}

void WorkerQueue::runThread(){
  while(processFront());
}


}}
