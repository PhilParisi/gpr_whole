#include "hp_optimizer.h"

namespace gpr {

class GprCostFunction : public ceres::FirstOrderFunction {
 public:
  virtual ~GprCostFunction() {}

  gpr::GprParams params;
//  CudaBlockMat<float> training_input;
//  CudaBlockMat<float> training_output;
//  CudaBlockMat<float> training_sensor_var;

  ss::bpslam::ParticlePtr_t particle;

  virtual bool Evaluate(const double* parameters,
                        double* cost,
                        double* gradient) const {


    //BlockGpr test_gpr;
    //test_gpr.params = params;

    ss::bpslam::GPRWorker worker(particle,params);
    worker.setupGPR();

//    // go through all the hyperparameters for our kernel and set them to our x test point
//    for(auto key_value : params.regression->kernel->getHPIndexMap()){
//      std::string key = key_value.first;
//      size_t index = key_value.second;
//      params.regression->kernel->hyperparam(key) = float( parameters[index] );
//    }
//    test_gpr.params.regression->kernel->hyperparam2dev();
//    for(size_t i = 0; i < training_input.nnz() ; i++){
//      test_gpr.addTrainingData(training_input.getBlock(i),training_output.getBlock(i));
//    }

    worker.addBlocks2GPR(particle);

    cost[0] = -particle->getData()->map.gpr->lml();
    if(std::isnan(cost[0])){
      return false;
      particle->getData()->map.gpr.reset();
    }
    std::cout << "residual    " <<  cost[0] << ", l " << parameters[0] << ", sigma " << parameters[1] << std::endl;


    if (gradient != NULL) {
          gradient[0] = - particle->getData()->map.gpr->derivativeLML(0);
          gradient[1] = - particle->getData()->map.gpr->derivativeLML(1);
    }
    std::cout << "  Partials:  l " << gradient[0] << ", sigma " << gradient[1] << std::endl;
    particle->getData()->map.gpr.reset();
    return true;


//    const double x = parameters[0];
//    const double y = parameters[1];
//    cost[0] = (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x);
//    if (gradient != NULL) {
//      gradient[0] = -2.0 * (1.0 - x) - 200.0 * (y - x * x) * 2.0 * x;
//      gradient[1] = 200.0 * (y - x * x);
//    }
//    return true;


  }
  virtual int NumParameters() const { return 2; }
};


HPOptimizer::HPOptimizer()
{
  return;
}

void HPOptimizer::optimize(gpr::GprParams params, ss::bpslam::ParticlePtr_t particle){
  double parameters[2] = {0,0};
  for(size_t i = 0 ; i<num_hp ; i++){
    parameters[i]=params.regression->kernel->hyperparam(i);
  }
  ceres::GradientProblemSolver::Options options;
  options.minimizer_progress_to_stdout = true;
  //options.min_line_search_step_size = 0.01;
  //options.function_tolerance = 100;
  options.line_search_type = ceres::ARMIJO;
  options.line_search_direction_type = ceres::STEEPEST_DESCENT;

  ceres::GradientProblemSolver::Summary summary;

  GprCostFunction * gpr_cost_funtion = new GprCostFunction;

  gpr_cost_funtion->params = params;
  gpr_cost_funtion->particle = particle;

  ceres::GradientProblem problem(gpr_cost_funtion);
  ceres::Solve(options, problem, parameters, &summary);
  std::cout << summary.FullReport() << "\n";
  std::cout << "Initial x: " << -1.2 << " y: " << 1.0 << "\n";
  std::cout << "Final   x: " << parameters[0]
            << " y: " << parameters[1] << "\n";
  //delete gpr_cost_funtion;

}

/*

class GprCostFunction : public ceres::SizedCostFunction<1, 2> {
 public:
  virtual ~GprCostFunction() {}

  gpr::GprParams params;
  CudaBlockMat<float> training_input;
  CudaBlockMat<float> training_output;
  CudaBlockMat<float> training_sensor_var;

  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
      BlockGpr test_gpr;
      test_gpr.params = params;

      // go through all the hyperparameters for our kernel and set them to our x test point
      for(auto key_value : test_gpr.params.regression->kernel->getHPIndexMap()){
        std::string key = key_value.first;
        size_t index = key_value.second;
        test_gpr.params.regression->kernel->hyperparam(key) = float( parameters[index][0] );
      }
      test_gpr.params.regression->kernel->hyperparam2dev();
      for(size_t i = 0; i < training_input.nnz() ; i++){
        test_gpr.addTrainingData(training_input.getBlock(i),training_output.getBlock(i));
      }
      residuals[0] = -test_gpr.lml();
      if(isnan(residuals[0]))
        return false;
      std::cout << "residual " <<  residuals[0] << ", l " << parameters[1][0] << ", sigma " << parameters[1][0] << std::endl;

      if (jacobians != NULL && jacobians[0] != NULL) {
            jacobians[0][0] = - test_gpr.derivativeLML(0);
            jacobians[1][0] = - test_gpr.derivativeLML(1);
      }

      return true;
  }
};


//bool HPCostFunction::Evaluate(const double *const *parameters, double *residual, double **jacobians) const{
//  BlockGpr test_gpr;
//  test_gpr.params = params;

//  // go through all the hyperparameters for our kernel and set them to our x test point
//  for(auto key_value : test_gpr.params.regression->kernel->getHPIndexMap()){
//    std::string key = key_value.first;
//    size_t index = key_value.second;
//    test_gpr.params.regression->kernel->hyperparam(key) = float( parameters[index][0] );
//  }
//  test_gpr.params.regression->kernel->hyperparam2dev();
//  for(size_t i = 0; i < training_input.nnz() ; i++){
//    test_gpr.addTrainingData(training_input.getBlock(i),training_output.getBlock(i));
//  }
//  residual[0] = -test_gpr.lml();
//  if(isnan(residual[0]))
//    return false;
//  std::cout << "residual " <<  residual[0] << ", l " << parameters[0][0] << ", sigma " << parameters[1][0] << std::endl;

//  if (jacobians != NULL && jacobians[0] != NULL) {
//        jacobians[0][0] = - test_gpr.derivativeLML(0);
//        jacobians[1][0] = - test_gpr.derivativeLML(1);
//  }

//  return true;
//}


HPOptimizer::HPOptimizer()
{
  return;
}

void HPOptimizer::optimize(HPCostFunctor::Ptr functor){
  //google::InitGoogleLogging(argv[0]);
  // The variable to solve for with its initial value. It will be
  // mutated in place by the solver.
  double arr[num_hp];
  for(size_t i = 0 ; i<num_hp ; i++){
    arr[i]=functor->params.regression->kernel->hyperparam(i);
  }
  double *x = arr;
  //const double initial_x[num_hp] = arr;

  // Build the problem.
  ceres::Problem problem;

  // Set up the only cost function (also known as residual). This uses
  // numeric differentiation to obtain the derivative (jacobian).
//  ceres::CostFunction* cost_function =
//      new ceres::NumericDiffCostFunction<HPCostFunctor, ceres::CENTRAL, 1, num_hp> (functor.get());
  std::shared_ptr<GprCostFunction> gpr_cost_funtion(new GprCostFunction);

  gpr_cost_funtion->params = functor->params;
  gpr_cost_funtion->training_input = functor->training_input;
  gpr_cost_funtion->training_output = functor->training_output;
  gpr_cost_funtion->training_sensor_var = functor->training_sensor_var;
  ceres::CostFunction* cost_function = gpr_cost_funtion.get();
//        gpr::GprParams params;
//        CudaBlockMat<float> training_input;
//        CudaBlockMat<float> training_output;
//        CudaBlockMat<float> training_sensor_var;
  problem.AddResidualBlock(cost_function, NULL, x);

  // Run the solver!
  ceres::Solver::Options options;
  options.max_num_iterations = 5000;
  options.minimizer_type = ceres::LINE_SEARCH;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << "\n";
  std::cout << "length-scale : "
            << " -> " << x[0] << "\n";
  std::cout << "sigma2e : "
            << " -> " << x[1] << "\n";

}

*/
}
