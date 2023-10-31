#include "../include/sensor_stream_ros/multires_pcl_gpr.h"
#include "ceres/ceres.h"
#include "glog/logging.h"

/*!
 *  this program was designed as a test to determine if ceres
 * sovler would work as a method to optimize my gpr hyperparameters
 */

using ceres::NumericDiffCostFunction;
using ceres::CENTRAL;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

struct CostFunctor {
  bool operator()(const double* const x, double* residual) const {


    PredParams itter_fine_param;
    itter_fine_param = fineParam;

    itter_fine_param.hyperParm.reset(2,1);
    itter_fine_param.hyperParm.initHost();
    itter_fine_param.hyperParm(0,0)= x[0];// 0.6f;  //  length scale
    itter_fine_param.hyperParm(1,0)= 1.0f;    //  alpha
    itter_fine_param.hyperParm.host2dev();
    itter_fine_param.sigma2e = x[1];// 2.0f;

    MultiresPclGpr predictor;

    predictor.readCloud(input_file);
    predictor.setOutputDir(output_dir);
    predictor.setParams(corseParam,itter_fine_param);
    predictor.predictTile(91.0,-23.2);

    residual[0] = - predictor.finePredictor.getGpr()->prediction.LML;
    std::cout << "residual " <<  residual[0] << ", l " << x[0] << ", sigma " << x[1] << std::endl;
    return true;
  }
  std::string input_file;
  std::string output_dir;
  PredParams fineParam;
  PredParams corseParam;

};

int main(int argc, char **argv){

  PredParams corseParam;
  corseParam.tileDim = 8;
  corseParam.trainingDim = 10;
  corseParam.divx = 400;
  corseParam.divy = 400;

  corseParam.hyperParm.reset(2,1);
  corseParam.hyperParm.initHost();
  corseParam.hyperParm(0,0)= 3.0f;  //  length scale
  corseParam.hyperParm(1,0)= 1.0f;    //  alpha
  corseParam.hyperParm.host2dev();
  corseParam.blockSize = 1200;
  corseParam.sigma2e = 1.4f;
  corseParam.randomSampleFactor = 0.05f;
  corseParam.maxPointsPerTile = 40000;

  PredParams fineParam;
  fineParam.tileDim = 8;
  fineParam.trainingDim = 3;
  fineParam.divx = 100;
  fineParam.divy = 100;

  fineParam.hyperParm.reset(2,1);
  fineParam.hyperParm.initHost();
  fineParam.hyperParm(0,0)= 0.6f;  //  length scale
  fineParam.hyperParm(1,0)= 1.0f;    //  alpha
  fineParam.hyperParm.host2dev();
  fineParam.blockSize = 1200;
  fineParam.sigma2e = 2.0f;
  fineParam.randomSampleFactor = -1.0f;
  fineParam.maxPointsPerTile = 30000;



  CostFunctor * functor = new CostFunctor;

  functor->corseParam = corseParam;
  functor->fineParam = fineParam;
  functor->input_file = argv[1];
  functor->output_dir = argv[2];



  google::InitGoogleLogging(argv[0]);

  // The variable to solve for with its initial value. It will be
  // mutated in place by the solver.
  double arr[2] = {0.6,2.0};
  double *x = arr;
  const double initial_x[2] = {0.6,2.0};

  // Build the problem.
  Problem problem;

  // Set up the only cost function (also known as residual). This uses
  // numeric differentiation to obtain the derivative (jacobian).
  CostFunction* cost_function =
      new NumericDiffCostFunction<CostFunctor, CENTRAL, 1, 2> (functor);
  problem.AddResidualBlock(cost_function, NULL, x);

  // Run the solver!
  Solver::Options options;
  options.max_num_iterations = 5000;
  options.minimizer_type = ceres::LINE_SEARCH;
  options.minimizer_progress_to_stdout = true;
  Solver::Summary summary;
  Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << "\n";
  std::cout << "length-scale : " << initial_x[0]
            << " -> " << x[0] << "\n";
  std::cout << "sigma2e : " << initial_x[1]
            << " -> " << x[1] << "\n";
  return 0;


    return 0;
}
