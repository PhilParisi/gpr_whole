#ifndef HP_OPTIMIZER_H
#define HP_OPTIMIZER_H

#include <sensor_stream/include/sensor_stream/blockgpr.h>
#include <sensor_stream/include/sensor_stream/gpr/fixedhpkernel.h>
#include "ceres/ceres.h"
#include "glog/logging.h"
#include <sensor_stream_ros/bpslam/worker/gpr_worker.h>


namespace gpr {

struct HPCostFunctor {
  typedef std::shared_ptr<HPCostFunctor> Ptr;
  bool operator()(const double* const x, double* residual) const;
  gpr::GprParams params;
  CudaBlockMat<float> training_input;
  CudaBlockMat<float> training_output;
  CudaBlockMat<float> training_sensor_var;

};

class HPCostFunction : public ceres::SizedCostFunction<1,2>{
public:
  virtual ~HPCostFunction() {}
  virtual bool Evaluate(double const* const* parameters,
                          double* residual,
                          double** jacobians) const;
  gpr::GprParams params;
  CudaBlockMat<float> training_input;
  CudaBlockMat<float> training_output;
  CudaBlockMat<float> training_sensor_var;
};

class HPOptimizer
{
public:
  static const int    num_hp = 2;
  HPOptimizer();
  void optimize(gpr::GprParams params, ss::bpslam::ParticlePtr_t particle);
  gpr::GprParams gpr_params_;
protected:

};

}



#endif // HP_OPTIMIZER_H
