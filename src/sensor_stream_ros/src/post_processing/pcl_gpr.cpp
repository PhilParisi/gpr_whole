#include "../include/sensor_stream_ros/multires_pcl_gpr.h"
#include <boost/program_options.hpp>

using namespace boost::program_options;

int main(int argc, char **argv){

  options_description desc{"Options"};
  desc.add_options()
    ("help,h"  , "Help screen")
    ("input,i" , value<std::string>(), "input pointcloud file")
    ("output,o", value<std::string>(), "output directory")
    ("config,c", value<std::string>(), "configuration YAML file");


  variables_map vm;
  store(parse_command_line(argc, argv, desc), vm);
  notify(vm);

  PredParams pred_params;
  pred_params.tileDim = 500;
  pred_params.trainingDim = 201;
  pred_params.divx = 50;
  pred_params.divy = 50;

  pred_params.hyperParm.reset(2,1);
  pred_params.hyperParm.initHost();
  pred_params.hyperParm(0,0)= 200;//15.0f;  //  length scale
  pred_params.hyperParm(1,0)= 50;//1.0f;    //  alpha
  pred_params.hyperParm.host2dev();
  pred_params.blockSize = 800;
  pred_params.sigma2e = 8;//1.4f;
  pred_params.randomSampleFactor = -1.0f;
  pred_params.maxPointsPerTile = 90000;
  pred_params.gridmap_mean_fn = true;
  pred_params.gridmap_res = 250.0;


  pclPredictor predictor;

  YAML::Node params;
  params["params"]=pred_params.toYaml();
  std::string outdir = vm["output"].as<std::string>();
  std::string saveCfg = outdir+"/config.yaml";
  std::ofstream fout(saveCfg);
  fout << params;

  predictor.setPredParams(pred_params);
  predictor.readCloud(vm["input"].as<std::string>());
  predictor.setOutputDir(vm["output"].as<std::string>());
  predictor.autoTile();


  return 0;
}
