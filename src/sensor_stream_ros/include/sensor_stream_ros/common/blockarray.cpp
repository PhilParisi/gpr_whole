#include "blockarray.h"
namespace ss{


ss::BlockArray::BlockArray()
{
//  block_params_.reset(new BlockParams);
//  block_params_->size=800;

}

void BlockArray::addEmptyBlock(){
  SensorFrameBlock::Ptr empty(new SensorFrameBlock(params.block_params));
  push_back(empty);
}

void ss::BlockArray::addPing(sensor_msgs::PointCloud2::ConstPtr ping){
  if(partial_.size()!=params.divisions){
    partial_.resize(params.divisions);
  }
  pcl::PointCloud<pcl::PointXYZI> pcl_cloud;
  pcl::fromROSMsg(*ping, pcl_cloud);
  size_t cloud_size = pcl_cloud.width * pcl_cloud.height;
  size_t chunk_size = cloud_size/(params.divisions);
  for (size_t swath_num = 0; swath_num<params.divisions ; swath_num++) {
    if(partial_[swath_num]==nullptr){
      partial_[swath_num].reset(new SensorFrameBlock(params.block_params));
    }
    partial_[swath_num]->addPing(ping->header);
    partial_[swath_num]->swath=swath_num;
    if(swath_num == params.divisions-1){
      for(size_t point_idx = swath_num*chunk_size; point_idx < cloud_size; point_idx++){
        if(partial_[swath_num]->isFull()){
          push_back(partial_[swath_num]);
          partial_[swath_num].reset(new SensorFrameBlock(params.block_params));
          partial_[swath_num]->addPing(ping->header);
        }else {
          partial_[swath_num]->addPoint(pcl_cloud[point_idx]);
        }
      }
    }else{
      for(size_t point_idx = swath_num*chunk_size; point_idx < swath_num*chunk_size+chunk_size; point_idx++){
        if(partial_[swath_num]->isFull()){
          push_back(partial_[swath_num]);
          partial_[swath_num].reset(new SensorFrameBlock(params.block_params));
          partial_[swath_num]->addPing(ping->header);
        }else {
          partial_[swath_num]->addPoint(pcl_cloud[point_idx]);
        }
      }
    }
  }
}

}
