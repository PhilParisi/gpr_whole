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

  PredParams corseParam;
  corseParam.tileDim = 3;
  corseParam.trainingDim = 15;
  corseParam.divx = 30;
  corseParam.divy = 30;

  corseParam.hyperParm.reset(2,1);
  corseParam.hyperParm.initHost();
  corseParam.hyperParm(0,0)= 15.0f;  //  length scale
  corseParam.hyperParm(1,0)= 1.0f;    //  alpha
  corseParam.hyperParm.host2dev();
  corseParam.blockSize = 800;
  corseParam.sigma2e = 1.4f;
  corseParam.randomSampleFactor = 0.05f;
  corseParam.maxPointsPerTile = 40000;
//  corseParam.gridmap_mean_fn = true;
//  corseParam.gridmap_res = 5.0;

  PredParams fineParam;
  fineParam.tileDim = 3;
  fineParam.trainingDim = 3;
  fineParam.divx = 30;
  fineParam.divy = 30;

  fineParam.hyperParm.reset(2,1);
  fineParam.hyperParm.initHost();
  fineParam.hyperParm(0,0)= 3.0;  //  length scale
  fineParam.hyperParm(1,0)= 0.02;    //  alpha
  fineParam.hyperParm.host2dev();
  fineParam.blockSize = 800;
  fineParam.sigma2e = 0.4f;
  fineParam.randomSampleFactor = //-1.0f;
  fineParam.maxPointsPerTile = 10000;
  MultiresPclGpr predictor;



  YAML::Node params;
  params["fine"]=fineParam.toYaml();
  params["corse"]=corseParam.toYaml();
  std::string outdir = vm["output"].as<std::string>();
  std::string saveCfg = outdir+"/config.yaml";
  std::ofstream fout(saveCfg);
  fout << params;


  predictor.readCloud(vm["input"].as<std::string>());
  predictor.setOutputDir(vm["output"].as<std::string>());
  predictor.setParams(corseParam,fineParam);
  predictor.autoTile();
//  predictor.predictTile(-20.1,-38.1);
//  std::cout << predictor.finePredictor.getGpr()->lml() << std::endl;
//  std::cout << predictor.finePredictor.getGpr()->derivativeLML(0) << std::endl;
//  std::cout << predictor.finePredictor.getGpr()->derivativeLML(1) << std::endl;

  return 0;
}
