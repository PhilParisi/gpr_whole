#include "gpr_mapper.h"

void GprMapperConfig::getFromServer(ros::NodeHandlePtr node){
  node->param<std::string>("map_frame",mapFrame,"map");
  node->param<std::string>("cloud_topic",inCloudTopic,"detections");
  node->param("swath_divisions",swathDivisions,4);
  node->param<float>("viz_rate",vizRate,10);
  node->param<float>("max_chol_size",max_chol_size,8.0);

  tile.reset(new ss::ros::TileParams);
  node->param<float>("tile/x_dim",tile->xdim,30.0f);
  node->param<float>("tile/y_dim",tile->ydim,tile->xdim);
  node->param<float>("tile/x_margin",tile->xMargin,5.0f);
  node->param<float>("tile/y_margin",tile->yMargin,tile->xMargin);

  block.reset(new ss::BlockParams);
  int blockSize;
  node->param<int>("block/size",blockSize,200);
  block->size=size_t(blockSize);
  if(blockSize%4 != 0)
    ROS_WARN("block_size param is not divisible by 4.  Tensor cores will be deactivated!");
}

void GprMapperConfig::writeToYaml(std::string filename){
  YAML::Node params;
  params["map_frame"]=mapFrame;
  params["cloud_topic"]=inCloudTopic;
  params["swath_divisions"]=swathDivisions;
  params["viz_rate"]=vizRate;
  params["tile/x_dim"]=tile->xdim;
  params["tile/y_dim"]=tile->ydim;
  params["tile/x_margin"]=tile->xMargin;
  params["tile/y_margin"]=tile->yMargin;
  params["block/size"]=block->size;
  std::ofstream fout(filename);
  fout << params;
}

GprMapper::GprMapper()
{
  _node.reset(new ros::NodeHandle);
  // setup the params from the server
  cfg.getFromServer(_node);

  //init variables
  _inputBlocks.resize(size_t(cfg.swathDivisions+1));
  _blockCount=0;
  _tiler.setTileParams(cfg.tile);
  _tiler.setBlockParams(cfg.block);

  // setup subs/pubs
  _inCloudSub = _node->subscribe<sensor_msgs::PointCloud2>(cfg.inCloudTopic,10000, &GprMapper::inCloudCallback, this);
  _tfListener = new  tf2_ros::TransformListener(_tfBuffer);
  _blockCloudPub = _node->advertise<sensor_msgs::PointCloud2>("viz/block_cloud",1);
  _predCloudPub = _node->advertise<sensor_msgs::PointCloud2>("viz/pred_cloud",1);
  _imTrans.reset(new image_transport::ImageTransport(*_node));
  _sparsityPub = _imTrans->advertise("viz/sparsity",1);


  if(cfg.vizRate>0){
    _vizTimer = _node->createTimer(ros::Duration(cfg.vizRate),&GprMapper::vizTimerCallback,this);
  }
  cfg.gpr.regression.reset(new gpr::RegressionParams);
  cfg.gpr.prediction.reset(new gpr::PredParams);
  cfg.gpr.regression->kernel.reset(new ss::kernels::SqExpSparse2d);
  cfg.gpr.regression->kernel->hyperparam("length_scale")  = 4.0f;
  cfg.gpr.regression->kernel->hyperparam("process_noise") = 0.1f;
  cfg.gpr.regression->kernel->hyperparam2dev();


//  cfg.gpr.regression->kernel=&gpr::kernels::sqExpSparse2d;
//  cfg.gpr.regression->hyperParam.reset(2,1);
//  cfg.gpr.regression->hyperParam.initHost();
//  cfg.gpr.regression->hyperParam(0,0)=.6f;
//  cfg.gpr.regression->hyperParam(1,0)=1.0f;
//  cfg.gpr.regression->hyperParam.host2dev();
  cfg.gpr.regression->sensor_var = powf(0.2f,2);
  cfg.gpr.regression->nnzThresh=1e-16f;
  ros::Duration(2).sleep();

}


void GprMapper::inCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& inCloud){
    try{
        sensor_msgs::PointCloud2 mapCloud;
        pcl::PointCloud<pcl::PointXYZI> pclCloud;
        mapCloud = _tfBuffer.transform(*inCloud,cfg.mapFrame);
        pcl::fromROSMsg (mapCloud, pclCloud);
        size_t cloud_size = mapCloud.width * mapCloud.height;
        size_t pointNum=0;
        int swathNum = -1;
        size_t chunckSize = cloud_size/(cfg.swathDivisions+1);
        for (pcl::PointCloud<pcl::PointXYZI>::const_iterator it = pclCloud.begin(); it != pclCloud.end(); ++it){
          pcl::PointXYZI pt = *it;
          if(pointNum%chunckSize==0){
              swathNum++;
              if(swathNum>cfg.swathDivisions)
                  swathNum=cfg.swathDivisions;
          }
          pt.intensity = swathNum;
          addToBlock(pt,swathNum);

          pointNum++;
        }
    }
    catch(tf2::ExtrapolationException e){
      return;
    }
    catch(tf2::LookupException e){
      std::cout<< e.what() << std::endl;
      return;
    }

}

void GprMapper::addToBlock(pcl::PointXYZI pt, size_t swathNum){

    _inputBlocks[swathNum].push_back(pt);
    if(_inputBlocks[swathNum].size()>=cfg.block->size){
        sensor_msgs::PointCloud2 cloudMsg;
        pcl::toROSMsg(*_inputBlocks[swathNum].getCloud(),cloudMsg);
        cloudMsg.header.stamp = ros::Time::now();
        cloudMsg.header.frame_id = cfg.mapFrame;
        _blockCloudPub.publish(cloudMsg);
        //ROS_INFO("block size = %i",_inputBlocks[swathNum].size());
        addBlockToGpr(swathNum);
        ss::SingleFrameBlock newBlock;
        _inputBlocks[swathNum]=newBlock;
    }
}

void GprMapper::addBlockToGpr(size_t swathNum){
    //ROS_INFO("adding block from swath %i ", swathNum);
    std::vector<ss::ros::Tile> visitedTiles;
    visitedTiles = _tiler.addBlock(_inputBlocks[swathNum]);

    for(size_t i = 0; i<visitedTiles.size() ; i++){
        if(!visitedTiles[i].isQueued()){
            visitedTiles[i].setQueued(true);
            _tileQueue.push_back(visitedTiles[i]);
        }
    }
}

void GprMapper::predict(float xCenter, float yCenter){
    pcl::PointCloud<pcl::PointXYZI> publishCloud;
    ROS_INFO("this tile has %i clouds ",_tiler.getTile(xCenter,yCenter).size());
    for(size_t i = 0 ; i<_tiler.getTile(xCenter,yCenter).size(); i++){
        pcl::PointCloud<pcl::PointXYZI>::Ptr blockCloud = _tiler.getTile(xCenter,yCenter).getBlock(i).getCloud();
        publishCloud += *blockCloud;
        //ROS_INFO("predicting %i ", i);
    }

//    ROS_INFO("predicting in the neighborhood of %f, %f", xCenter, yCenter);
//    ROS_INFO("cloud size %i ", publishCloud.size());
    sensor_msgs::PointCloud2 cloudMsg;
    pcl::toROSMsg(publishCloud,cloudMsg);
    cloudMsg.header.stamp = ros::Time::now();
    cloudMsg.header.frame_id = cfg.mapFrame;

    _predCloudPub.publish(cloudMsg);
}

void GprMapper::doTile(ss::ros::Tile workingTile){
  float chol_mem = workingTile.getData()->cholNnz * pow(cfg.block->size,2) * sizeof (float) * 1e-9;
  if(chol_mem>cfg.max_chol_size){
      ROS_WARN("This tile has already exceeded max_chol_size param (%f) nnz: %zu, size: %f Gb,", cfg.max_chol_size, gpr->getCholeskyFactor().nnz(), chol_mem);
      return;
  }
  //BlockGpr gpr;
  gpr.reset(new BlockGpr);
  gpr->params=cfg.gpr;

  // center prediction around average z of tile

  float offset=0;
  offset=workingTile.getAvgZ();

  for (size_t i = 0; i < workingTile.size(); i++) {
      chol_mem = workingTile.getData()->cholNnz * pow(cfg.block->size,2) * sizeof (float) * 1e-9;
      if(chol_mem>cfg.max_chol_size){
          ROS_WARN("Chol is greater than max_chol_size param (%f) nnz: %zu, size: %f Gb,", cfg.max_chol_size, gpr->getCholeskyFactor().nnz(), chol_mem);
          break;
      }
      ss::TrainingBlock trainingData = workingTile.getBlock(i).getTrainingData();
      add(trainingData.y,-offset);
      gpr->addTrainingData(trainingData.x,trainingData.y);
      workingTile.getData()->cholNnz = gpr->getCholeskyFactor().nnz();
      ros::spinOnce();

  }

  ss::ros::TileParams tileParam = workingTile.getTileParam();
  ss::ros::TileDataPtr tileData = workingTile.getData();

  gpr->predict(tileData->xOrigin-tileParam.xdim/2,
               tileData->xOrigin+tileParam.xdim/2,
               tileData->yOrigin-tileParam.ydim/2,
               tileData->yOrigin+tileParam.ydim/2,
               75,75);

  add(gpr->prediction.mu,offset);
  gpr->prediction.points.dev2host();
  gpr->prediction.mu.dev2host();
  gpr->prediction.sigma.dev2host();

  pcl::PointCloud<pcl::PointXYZI>::Ptr outCloud (new pcl::PointCloud<pcl::PointXYZI>);
  outCloud->width  = gpr->prediction.mu.size();
  outCloud->height  = 1;
  outCloud->points.resize (gpr->prediction.mu.size());

  gpr->prediction.dev2host();
  for(size_t i = 0 ; i<outCloud->size() ; i++){
    //if(gpr->prediction.sigma(i,0)>5){
      outCloud->points[i].x=gpr->prediction.points(i,0);
      outCloud->points[i].y=gpr->prediction.points(i,1);
      if(std::isnan(gpr->prediction.mu(i,0)))
          outCloud->points[i].z=0;
      else
          outCloud->points[i].z=gpr->prediction.mu(i,0);
      outCloud->points[i].intensity=log( gpr->prediction.sigma(i,0) );


    //}

  }
  sensor_msgs::PointCloud2::Ptr cloudMsg;
  cloudMsg.reset(new sensor_msgs::PointCloud2);
  pcl::toROSMsg(*outCloud,*cloudMsg);
  cloudMsg->header.stamp = ros::Time::now();
  cloudMsg->header.frame_id = cfg.mapFrame;



  workingTile.setPrediction(cloudMsg);
  _predCloudPub.publish(workingTile.getPrediction());

  publishSparsity();
}

void GprMapper::publishSparsity(){
    if(_sparsityPub.getNumSubscribers()>0){
        cv::Mat sparsity = ss::viz::sparsity2image(gpr->getCholeskyFactor(),2.0f*float(cfg.block->size)/float(100));
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", sparsity).toImageMsg();
        _sparsityPub.publish(msg);
    }
}



void GprMapper::spinOnce(){
    // process the front of the queue
    if(_tileQueue.size()>0){  // bail out if queue is empty
        doTile(_tileQueue.front());
        _tileQueue.front().setQueued(false);
        _tileQueue.pop_front();
    }
    ros::spinOnce();
}

void GprMapper::spin(){
    while(ros::ok()){
        spinOnce();
    }
}

void GprMapper::vizTimerCallback(const ros::TimerEvent &){
    std::thread t(&GprMapper::publishPredictions,this);
    t.detach();
}

void GprMapper::publishPredictions(){
    ROS_INFO("publishing predictions");
    for(size_t i = 0 ; i< _tiler.getTileList().size() ; i++){
      _tiler.tileList(i).getData()->mutex.lock();
      if(_tiler.tileList(i).getPrediction()!=nullptr){
        _predCloudPub.publish(*_tiler.tileList(i).getPrediction());
      }
      _tiler.tileList(i).getData()->mutex.unlock();
      ros::Duration(.01).sleep();
    }
    ROS_INFO("finished publishing");
}
