#include "pcl_predictor.h"

PredParams::PredParams(){
  gridmap_mean_fn = false;
  gridmap_res = 100;
}

YAML::Node PredParams::toYaml(){
  YAML::Node params;
  params["tile/x_dim"]=tileDim;
  params["tile/x_margin"]=trainingDim;
  params["block/size"]=blockSize;
  params["sigma2e"]=sigma2e;
  std::vector<float> hpVect;
  for(size_t i = 0 ; i<hyperParm.size(); i++)
    hpVect.push_back(hyperParm(i,0));
  params["hyper_param"]=hpVect;
  params["divx"]=divx;
  params["divy"]=divy;
  params["random_sample_factor"]=randomSampleFactor;
  params["max_points_per_tile"]=maxPointsPerTile;
  params["gridmap_mean_fn"]=gridmap_mean_fn;
  return params;
}
void PredParams::writeToYaml(std::string fName){
  boost::filesystem::path filename(fName);
  if(!boost::filesystem::exists(filename.parent_path())){
    boost::filesystem::create_directory(filename.parent_path());
  }

  std::ofstream fout(fName);
  fout << toYaml();
}

void PredParams::readYaml(YAML::Node params){

//  params["tile/x_dim"] >> tileDim;
//  params["tile/x_margin"]=trainingDim;
//  params["block/size"]=blockSize;
//  params["sigma2e"]=sigma2e;
//  std::vector<float> hpVect;
//  for(size_t i = 0 ; i<hyperParm.hostVector().size(); i++)
//    hpVect.push_back(hyperParm.hostVector()[i]);
//  params["hyper_param"]=hpVect;
//  params["divx"]=divx;
//  params["divy"]=divy;
//  params["random_sample_factor"]=randomSampleFactor;
//  params["max_points_per_tile"]=maxPointsPerTile;

//  ui->origin_depth->setValue(config["origin_depth"].as<double>());
//  ui->origin_x->setValue(config["xoff"].as<double>());
//  ui->origin_y->setValue(config["yoff"].as<double>());
//  ui->origin_lat->setValue(config["origin_lat"].as<double>());
//  ui->origin_lon->setValue(config["origin_lon"].as<double>());
}

pclPredictor::pclPredictor()
{
    _tileIndex=-1;
    return;
}

bool pclPredictor::readCloud(std::string filename){
    _cloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
    if (pcl::io::loadPCDFile<pcl::PointXYZI> (filename, *_cloud) == -1) //* load the file
    {
        std::cerr<< "Couldn't read file " << filename << std::endl;
        return false;
    }else{
        std::cout << "the following file was loaded: " << filename << std::endl;
        std::cout << "it has size: " << _cloud->size() << std::endl;
        updateCloudDim();
        std::cout << "Max x: " << maxPt.x << std::endl;
        std::cout << "Max y: " << maxPt.y << std::endl;
        std::cout << "Max z: " << maxPt.z << std::endl;
        std::cout << "Min x: " << minPt.x << std::endl;
        std::cout << "Min y: " << minPt.y << std::endl;
        std::cout << "Min z: " << minPt.z << std::endl;

        if(_predParams.gridmap_mean_fn){
            grid_map_.reset(new grid_map::GridMap({"elevation","N","SUM"}));
            grid_map_->setFrameId("map");
            // I had to add a small extra margin (10%) otherwise points at the edges would be excluded
            grid_map::Length len(abs(maxPt.x-minPt.x)+2*_predParams.gridmap_res,abs(maxPt.x-minPt.x)+2*_predParams.gridmap_res);
            grid_map::Position pos(minPt.x+len.x()/2,minPt.y+len.y()/2);
            grid_map_->setGeometry(len, _predParams.gridmap_res, pos);

            printf("Created map with size %f x %f m (%i x %i cells). \n",
              grid_map_->getLength().x(), grid_map_->getLength().y(),
              grid_map_->getSize()(0), grid_map_->getSize()(1));
          for (size_t i = 0 ;i<_cloud->size(); i++){
            grid_map::Position position;
            position.x() = double(_cloud->points[i].x);
            position.y() = double(_cloud->points[i].y);
            try{
              if(std::isnan(grid_map_->atPosition("N", position))){
                  grid_map_->atPosition("N", position)=0;
              };
              if(std::isnan(grid_map_->atPosition("SUM", position))){
                  grid_map_->atPosition("SUM", position)=0;
              };
              if(std::isnan(grid_map_->atPosition("elevation", position))){
                  grid_map_->atPosition("elevation", position)=0;
              };

              grid_map_->atPosition("N", position) = grid_map_->atPosition("N", position) +1;
              grid_map_->atPosition("SUM", position) = grid_map_->atPosition("SUM", position) + _cloud->points[i].z;
              grid_map_->atPosition("elevation", position) = grid_map_->atPosition("SUM", position)/grid_map_->atPosition("N", position);
            }catch(std::out_of_range){
              printf("point (%f,%f) is outside the gridmap\n",_cloud->points[i].x,_cloud->points[i].y);
            }

          }
        }
      return true;
  }
}

void pclPredictor::subsampleInputCloud(float ratio){
  //_filteredCloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::RandomSample<pcl::PointXYZI> filter;
  filter.setSample(_cloud->size()*ratio);
  filter.setInputCloud(_cloud);
  filter.filter(*_cloud);
  //pcl::io::savePCDFileASCII("/home/kris/Dropbox/Ubuntu/Data/wiggles_bank/output/subsampled.pcd",*_cloud);
}

void pclPredictor::subsampleFilteredCloud(float points){
  if(_filteredCloud->size()>points){
    pcl::RandomSample<pcl::PointXYZI> filter;
    filter.setSample(points);
    filter.setInputCloud(_filteredCloud);
    filter.filter(*_filteredCloud);
  }
  //pcl::io::savePCDFileASCII("/home/kris/Dropbox/Ubuntu/Data/wiggles_bank/output/subsampled.pcd",*_cloud);
}

void pclPredictor::updateCloudDim(){
    pcl::getMinMax3D (*_cloud, minPt, maxPt);
}

void pclPredictor::solveNextTile(){
    _tileIndex++;
    _gpr.reset(new BlockGpr);
}

void pclPredictor::cloud2Gpr(){
  _gpr.reset(new BlockGpr);
  ss::kernels::SqExpSparse2d::Ptr kernel(new ss::kernels::SqExpSparse2d);
  kernel->setHyperparams(_predParams.hyperParm);
  _gpr->params.regression->kernel=kernel;  //&gpr::kernels::sqExpKernel2d;
  _gpr->params.regression->sensor_var=powf(_predParams.sigma2e,2);

  for (size_t i = 0 ;i<_filteredCloud->size(); ) { // i is the cloud index
    CudaMat<float> x(_predParams.blockSize,2);
    CudaMat<float> y(_predParams.blockSize,1);

    if(_predParams.gridmap_mean_fn){
      for (size_t j = 0; j<_predParams.blockSize && i<_filteredCloud->size() ; j++) { // j is the block index
        float offset = grid_map_->atPosition("elevation",
                                             grid_map::Position(_filteredCloud->points[i].x,_filteredCloud->points[i].y),
                                             grid_map::InterpolationMethods::INTER_LINEAR);
        x(j,0)=_filteredCloud->points[i].x;
        x(j,1)=_filteredCloud->points[i].y;
        y(j,0)=_filteredCloud->points[i].z-offset;
        i++;
      }
    }else{
      for (size_t j = 0; j<_predParams.blockSize && i<_filteredCloud->size() ; j++) { // j is the block index
        x(j,0)=_filteredCloud->points[i].x;
        x(j,1)=_filteredCloud->points[i].y;
        y(j,0)=_filteredCloud->points[i].z;
        i++;
      }
    }
    x.host2dev();
    y.host2dev();
    _gpr->addTrainingData(x,y);
  }
}

void pclPredictor::filter2Tile(float tileCenterX, float tileCenterY){
  _filteredCloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::CropBox<pcl::PointXYZI> boxFilter;
  float minX = tileCenterX - (_predParams.tileDim+_predParams.trainingDim)/2.0f;
  float minY = tileCenterY - (_predParams.tileDim+_predParams.trainingDim)/2.0f;
  float minZ = minPt.z;
  float maxX = tileCenterX + (_predParams.tileDim+_predParams.trainingDim)/2.0f;
  float maxY = tileCenterY + (_predParams.tileDim+_predParams.trainingDim)/2.0f;
  float maxZ = maxPt.z;
  boxFilter.setMin(Eigen::Vector4f(minX, minY, minZ, 1.0));
  boxFilter.setMax(Eigen::Vector4f(maxX, maxY, maxZ, 1.0));
  boxFilter.setInputCloud(_cloud);
  boxFilter.filter(*_filteredCloud);
  if(_filteredCloud->size() > _predParams.maxPointsPerTile){
    std::cout << "Warning!  Too many points in tile.   Approximating.";
  }
  subsampleFilteredCloud(_predParams.maxPointsPerTile);
}

void pclPredictor::predict(CudaMat<float> predGrid){
  cloud2Gpr();
  _gpr->predict(predGrid);
  generateCloud();
}

void pclPredictor::predict(float x1Min, float x1Max,float x2Min, float x2Max, size_t x1Div, size_t x2Div){
  cloud2Gpr();
  _gpr->predict( x1Min,  x1Max, x2Min,  x2Max,  x1Div,  x2Div);
  generateCloud();
}

void pclPredictor::predictTile(float tileCenterX, float tileCenterY){
  cloud2Gpr();
  float x1Min = tileCenterX - (_predParams.tileDim)/2.0f;
  float x1Max = tileCenterX + (_predParams.tileDim)/2.0f;
  float x2Min = tileCenterY - (_predParams.tileDim)/2.0f;
  float x2Max = tileCenterY + (_predParams.tileDim)/2.0f;
  _gpr->predict( x1Min,  x1Max, x2Min,  x2Max,  _predParams.divx,  _predParams.divy);
  generateCloud();
}


void pclPredictor::generateCloud(){
  _predictedCloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
  _predictedCloud->width  = _gpr->prediction.points.rows();
  _predictedCloud->height  = 1;
  _predictedCloud->points.resize (_predictedCloud->width*_cloud->height);
  _gpr->prediction.mu.dev2host();
  _gpr->prediction.sigma.dev2host();
  _gpr->prediction.points.dev2host();

  for(size_t i = 0 ; i<_predictedCloud->size() ; i++){
      _predictedCloud->points[i].x=_gpr->prediction.points(i,0);
      _predictedCloud->points[i].y=_gpr->prediction.points(i,1);
      if(std::isnan(_gpr->prediction.mu(i,0)))
          _predictedCloud->points[i].z=0;
      else{
        if(_predParams.gridmap_mean_fn){
          float offset;
          try {
            offset = grid_map_->atPosition("elevation",
                                               grid_map::Position(_gpr->prediction.points(i,0),_gpr->prediction.points(i,1)),
                                               grid_map::InterpolationMethods::INTER_LINEAR);
          } catch (std::out_of_range) {
            offset = 0;
          }

          _predictedCloud->points[i].z=_gpr->prediction.mu(i,0)+offset;
        }else{
          _predictedCloud->points[i].z=_gpr->prediction.mu(i,0);
        }
      }
      _predictedCloud->points[i].intensity=sqrt(_gpr->prediction.sigma(i,0));
  }
}

void pclPredictor::savePredCloud(std::string filename){
  //std::cout<< "saving... "<< std::endl;

  pcl::io::savePCDFileBinary(filename,*_predictedCloud);
}

void pclPredictor::autoTile(){
  size_t tileno=0;
  for (float i = minPt.x;i<maxPt.x;i+=getPredParams().tileDim) {
    for (float j = minPt.y;j<maxPt.y;j+=getPredParams().tileDim) {
      filter2Tile(i,j);
//      if(getFilteredCloud()->size()>1){
//        //return;
//      }
      std::cout << "predicting tile: " << i << "," << j << std::endl;
      if(getFilteredCloud()->size()>1){
        predictTile(i,j);
      }

      if(getPredCloud() && getPredCloud()->size()>0){
        std::string fname = output_dir+"tile"+std::to_string(tileno)+".pcd";
        savePredCloud(fname);
        tileno++;
      }
    }
  }
}



//void pclPredictor::predict(CudaMat<float> predGrid){
//    int outIndex = 0;
//    float trainingDim = _predParams.trainingDim;
//    float tileDim = _predParams.tileDim;
//    for(float tileMinX=minPt.x ; tileMinX<maxPt.x; tileMinX+=tileDim){
//        for(float tileMinY=minPt.y ; tileMinY<maxPt.y; tileMinY+=tileDim){
////            try{
//            BlockGpr testGP;
//            testGP.params.regression->sigma2e = _predParams.sigma2e;//
//            size_t samples = _cloud->size();
//            CudaMat<float> hp = _predParams.hyperParm;
//            testGP.setKernel();
//            testGP.setHyperParam(hp);
//            size_t divx = _predParams.divx;
//            size_t divy = _predParams.divy;
//            size_t blockSize=_predParams.blockSize;  //integer


//            float xMin,xMax,yMin,yMax;
//            xMin=tileMinX-trainingDim;
//            xMax=tileMinX+tileDim+trainingDim;
//            yMin=tileMinY-trainingDim;
//            yMax=tileMinY+tileDim+trainingDim;

//            for(size_t pointIndex=0;pointIndex < samples; pointIndex++){
//                size_t blockIndex=0;
//                CudaMat<float> x(blockSize,2);
//                CudaMat<float> train(blockSize,1);
//                while(pointIndex<samples && blockIndex<blockSize ){
//                    if(_cloud->points[pointIndex].x > xMin && _cloud->points[pointIndex].x < xMax &&
//                       _cloud->points[pointIndex].y > yMin && _cloud->points[pointIndex].y < yMax){
//                        x(blockIndex,0) = _cloud->points[pointIndex].x; // x point
//                        x(blockIndex,1) = _cloud->points[pointIndex].y; // y point
//                        train(blockIndex,0) = _cloud->points[pointIndex].z; // z point
//                        blockIndex++;
//                    }
//                    pointIndex++;
//                }
//                //if blockIndex<
//                x.host2dev();
//                train.host2dev();
//                testGP.addTrainingData(x,train);
//                printf("Tile no. %i \n",outIndex);
//            }
//            testGP.predict(predGrid);
//            //testGP.predict(tileMinX,tileMinX+tileDim,tileMinY,tileMinY+tileDim,divx,divy);
//            //--------------
//            // Create output cloud
//            //--------------
//            pcl::PointCloud<pcl::PointXYZI>::Ptr outCloud (new pcl::PointCloud<pcl::PointXYZI>);
//            outCloud->width  = predGrid.rows();
//            outCloud->height  = 1;
//            outCloud->points.resize (predGrid.rows());
//            testGP.prediction.points.dev2host();
//            testGP.prediction.mu.dev2host();
//            testGP.prediction.sigma.dev2host();

//            for(size_t i = 0 ; i<outCloud->size() ; i++){
//                outCloud->points[i].x=testGP.prediction.points(i,0);
//                outCloud->points[i].y=testGP.prediction.points(i,1);
//                if(std::isnan(testGP.prediction.mu(i,0)))
//                    outCloud->points[i].z=0;
//                else
//                    outCloud->points[i].z=testGP.prediction.mu(i,0);
//                outCloud->points[i].intensity=testGP.prediction.sigma(i,0);
//            }
//            std::cout<< "saving... "<< std::endl;
//            pcl::io::savePCDFileASCII("/data/testing/SensorStream/wiggles_bank/tiles/tile"+ std::to_string(outIndex)+".pcd",*outCloud);


//            outIndex++;

//        }
//    }
//}
