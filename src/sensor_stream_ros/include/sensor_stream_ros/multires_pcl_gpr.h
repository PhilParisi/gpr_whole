#ifndef MULTIRES_PCL_GPR_H
#define MULTIRES_PCL_GPR_H

#include "pcl_predictor.h"


class MultiresPclGpr
{
public:
  MultiresPclGpr();
  void setOutputDir(std::string dir){output_dir=dir;}
  void readCloud(std::string cloudFile){finePredictor.readCloud(cloudFile);
                                        corsePredictor.readCloud(cloudFile);}
  void setParams(PredParams corse,PredParams fine);
  void predictTile(float tileCenterX,float tileCenterY);
  void autoTile();

  pclPredictor finePredictor;
  pclPredictor corsePredictor;
protected:
  std::string output_dir;
};

#endif // MULTIRES_PCL_GPR_H
