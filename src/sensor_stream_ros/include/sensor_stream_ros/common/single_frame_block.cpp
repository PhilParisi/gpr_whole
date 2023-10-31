#include "single_frame_block.h"
namespace ss {


SingleFrameBlock::SingleFrameBlock(){
    clear();
}


SingleFrameBlock::SingleFrameBlock(std::shared_ptr<BlockParams> params){
    setParams(params);
    _cloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
}

SingleFrameBlock::SingleFrameBlock(BlockParams::Ptr params,pcl::PointCloud<pcl::PointXYZI>::Ptr cloud){
    setParams(params);
    setPointcloud(cloud);
}

void SingleFrameBlock::SingleFrameBlock::clear(){
    params_ = nullptr;
    _cloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
    centerOfMass.x = 0;
    centerOfMass.y = 0;
    centerOfMass.z = 0;
}

void SingleFrameBlock::SingleFrameBlock::setPointcloud(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud){
    size_t cloud_size = cloud->width * cloud->height;
    if(cloud_size!=getParams()->size)
        throw std::out_of_range("setPointcloud error:  cloud size != BlockParams size");

    _cloud=cloud;
    computeCenterOfMass();

    return;
}

void SingleFrameBlock::SingleFrameBlock::push_back(pcl::PointXYZI pt){
    _cloud->push_back(pt);
    centerOfMass.x += pt.x;
    centerOfMass.y += pt.y;
    centerOfMass.z += pt.z;
}

void SingleFrameBlock::SingleFrameBlock::computeCenterOfMass(){
    size_t cloud_size = _cloud->width * _cloud->height;
    float x = 0;
    float y = 0;
    float z = 0;
    for ( size_t i = 0; i < cloud_size; i++ )
    {
        x += _cloud->at(i).x;
        y += _cloud->at(i).y;
        z += _cloud->at(i).z;
    }
    centerOfMass.x = x;
    centerOfMass.y = y;
    centerOfMass.z = z;
}

pcl::PointXYZ SingleFrameBlock::SingleFrameBlock::getCenterOfMass(){
    size_t cloud_size = _cloud->width * _cloud->height;
    pcl::PointXYZ out;
    out.x = centerOfMass.x/cloud_size;
    out.y = centerOfMass.y/cloud_size;
    out.z = centerOfMass.z/cloud_size;
    return out;
}

TrainingBlock SingleFrameBlock::SingleFrameBlock::getTrainingData(){
    TrainingBlock block;

    block.x.reset(params_->size,2);
    block.x.initHost();

    block.y.reset(params_->size,1);
    block.y.initHost();

    size_t i = 0;
    for (pcl::PointCloud<pcl::PointXYZI>::const_iterator it = _cloud->begin(); it != _cloud->end(); ++it){
        block.x(i,0) = it->x;
        block.x(i,1) = it->y;
        block.y(i,0) = it->z;
        i++;
    }

    block.x.host2dev();
    block.y.host2dev();

    return block;
}

TrainingBlock SingleFrameBlock::getSubsampledTrainingData(size_t block_size){
  TrainingBlock block;
  pcl::PointCloud<pcl::PointXYZI> subsampled_cloud;

  block.x.reset(block_size,2);
  block.x.initHost();

  block.y.reset(block_size,1);
  block.y.initHost();


  pcl::RandomSample<pcl::PointXYZI> filter;
  filter.setSample(block_size);
  filter.setInputCloud(_cloud);
  filter.filter(subsampled_cloud);

  size_t i = 0;
  for (pcl::PointCloud<pcl::PointXYZI>::const_iterator it = subsampled_cloud.begin(); it != subsampled_cloud.end(); ++it){
      block.x(i,0) = it->x;
      block.x(i,1) = it->y;
      block.y(i,0) = it->z;
      i++;
  }

  block.x.host2dev();
  block.y.host2dev();

  return block;


}

}
