#ifndef PARTICLEWORKER_H
#define PARTICLEWORKER_H
#include <thread>
#include <mutex>
#include <deque>
#include <atomic>
#include <vector>
#include <iostream>

namespace ss{ namespace bpslam {
namespace cmd {
  enum QueueCmd{
    run,
    sync,
    wait
  };
}

class ParticleWorker
{
public:
  ParticleWorker();
  virtual void run()=0;
  cmd::QueueCmd cmd;
  typedef std::shared_ptr<ParticleWorker> Ptr;
};

class SyncCmd: public ParticleWorker{
public:
  SyncCmd(){cmd=cmd::sync;}
  void run(){return;}
};

class WaitCmd: public ParticleWorker{
public:
  WaitCmd(){cmd=cmd::wait;}
  void run(){return;}
};

class WorkerQueue{
public:
  typedef std::shared_ptr<WorkerQueue> Ptr;
  WorkerQueue();
  ~WorkerQueue();
  void pushBack(ParticleWorker::Ptr worker);
  void pushFront(ParticleWorker::Ptr worker);
  ParticleWorker::Ptr popFront();
  /*!
   * \brief process and pops the front of the queue.
   */
  bool processFront();
  void addSyncCommand();
  void sync();
  void run(int thread_count = -1);
private:
  void runThread();
  bool running_;
  std::mutex queue_mutex_;
  std::atomic<bool> ready2sync_;
  std::atomic<size_t> thread_count_;
  std::deque<ParticleWorker::Ptr> worker_queue_;
  std::vector<std::thread> threads_;
};

}}

#endif // PARTICLEWORKER_H
