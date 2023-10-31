#ifndef PERF_TIMER_H
#define PERF_TIMER_H
#include <chrono> // for std::chrono functions

class PerfTimer
{
private:
  // Type aliases to make accessing nested type easier
  using clock_t = std::chrono::high_resolution_clock;
  using second_t = std::chrono::duration<double, std::ratio<1> >;

  std::chrono::time_point<clock_t> m_beg;
  std::chrono::time_point<clock_t> paused_time_;
  double duration_paused;
  bool paused;


public:
  PerfTimer() : paused_time_(clock_t::now()),
                m_beg(clock_t::now())

  {
    duration_paused=0;
  }

  void reset()
  {
    m_beg = clock_t::now();
    duration_paused=0;
    paused = false;
  }

  void pause(){
    paused_time_ = clock_t::now();
    paused = true;
  }

  void resume(){
    if(paused){
      duration_paused += std::chrono::duration_cast<second_t>(clock_t::now() - paused_time_).count();
    }
    paused = false;
  }

  double elapsed() const
  {
    return std::chrono::duration_cast<second_t>(clock_t::now() - m_beg).count() - duration_paused;
  }
};

#endif // PERF_TIMER_H
