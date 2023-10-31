#ifndef PCL_PREDICTOR_H
#define PCL_PREDICTOR_H

#include <sensor_stream/blockgpr.h>
#include "block_tiler.h"
#include "ss_cv.h"

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/impl/pcd_io.hpp>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/filters/passthrough.h>
#include <memory>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/random_sample.h>
#include "yaml-cpp/yaml.h"
#include <sensor_stream/include/sensor_stream/gpr/sqexpsparse2d.h>
#include "grid_map_ros/grid_map_ros.hpp"


struct PredParams{
  PredParams();
    float tileDim;
    float trainingDim;
    float sigma2e;
    CudaMat<float> hyperParm;
    size_t divx;
    size_t divy;
    size_t blockSize;
    float randomSampleFactor;
    size_t maxPointsPerTile;
    bool gridmap_mean_fn;
    double gridmap_res;

  YAML::Node toYaml();
  void writeToYaml(std::string fName);
  void readYaml(YAML::Node params);
};

class pclPredictor
{
public:
  pclPredictor();
  bool readCloud(std::string filename);
  void subsampleInputCloud(float ratio);
  void subsampleFilteredCloud(float points);
  void updateCloudDim();
  void cloud2Gpr();
  void filter2Tile(float tileCenterX,float tileCenterY);
  void predict(CudaMat<float> predGrid);
  void predict(float x1Min, float x1Max,float x2Min, float x2Max, size_t x1Div, size_t x2Div);
  void predictTile(float tileCenterX,float tileCenterY);
  void savePredCloud(std::string filename);
  void solveNextTile();
  void predCurrentTile(CudaMat<float> predGrid);
  void setPredParams(PredParams params){_predParams = params;}
  PredParams & getPredParams(){return _predParams;}
  pcl::PointCloud<pcl::PointXYZI>::Ptr getCloud(){return _cloud;}
  pcl::PointCloud<pcl::PointXYZI>::Ptr getFilteredCloud(){return _filteredCloud;}
  pcl::PointCloud<pcl::PointXYZI>::Ptr getPredCloud(){return _predictedCloud;}
  pcl::PointXYZI minPt, maxPt;
  std::shared_ptr<BlockGpr> getGpr(){return _gpr;}
  void generateCloud();
  void autoTile();
  void setOutputDir(std::string dir){output_dir=dir;}

private:

  pcl::PointCloud<pcl::PointXYZI>::Ptr _cloud;
  pcl::PointCloud<pcl::PointXYZI>::Ptr _filteredCloud;
  pcl::PointCloud<pcl::PointXYZI>::Ptr _predictedCloud;
  PredParams _predParams;
  std::shared_ptr<BlockGpr> _gpr;
  int _tileIndex;
  float _tileMinX;
  float _tileMinY;
  std::string output_dir;
  std::shared_ptr<grid_map::GridMap> grid_map_;
};

#endif // PCL_PREDICTOR_H
