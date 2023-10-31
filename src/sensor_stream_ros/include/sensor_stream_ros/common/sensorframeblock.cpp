#include "sensorframeblock.h"
namespace ss{


SensorFrameBlock::SensorFrameBlock(BlockParams::Ptr params)
{
  setParams(params);
  size_=0;
}

bool SensorFrameBlock::isFull(){
  return  params_->size==size();
}

void SensorFrameBlock::addPoint(pcl::PointXYZI point){
  size_++;
  if(size_>params_->size){
   throw std::runtime_error("Too many points added to block");
  }
  buffer_cloud_->push_back(point);
}

void SensorFrameBlock::addPing(std_msgs::Header header){
  // push the old buffer into clouds_
  if(buffer_cloud_!=nullptr){
    sensor_msgs::PointCloud2::Ptr cloud_msg(new sensor_msgs::PointCloud2);
    pcl::toROSMsg(*buffer_cloud_,*cloud_msg);
    cloud_msg->header = buffer_header_;
    clouds_.push_back(cloud_msg);
  }
  //  reset the buffer and the header
  buffer_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>);
  buffer_header_=header;
}

size_t SensorFrameBlock::size(){
  return size_;
}

void SensorFrameBlock::clearBuffer(){
  // push the old buffer into clouds_
  std::lock_guard<std::mutex> lock(read_mtx_);
  if(buffer_cloud_!=nullptr){
    sensor_msgs::PointCloud2::Ptr cloud_msg(new sensor_msgs::PointCloud2);
    pcl::toROSMsg(*buffer_cloud_,*cloud_msg);
    cloud_msg->header = buffer_header_;
    clouds_.push_back(cloud_msg);

    //  clear the buffer and set to null
    buffer_cloud_.reset();
  }
}

SensorFrameBlock::Ptr SensorFrameBlock::transform(tf2_ros::Buffer & buffer, std::string frame){
  clearBuffer();
  SensorFrameBlock::Ptr out(new SensorFrameBlock(params_));

  for (auto ping : clouds_) {
    sensor_msgs::PointCloud2::Ptr transformed_ping(new sensor_msgs::PointCloud2);
    *transformed_ping = buffer.transform(*ping,frame);
    out->clouds_.push_back(transformed_ping);
  }
  out->swath=swath;
  return out;
}

SingleFrameBlock::Ptr SensorFrameBlock::transform2StaticFrame(tf2_ros::Buffer &buffer, std::string frame){
  clearBuffer();
  SingleFrameBlock::Ptr out(new SingleFrameBlock(params_));
  out->getCloud().reset(new pcl::PointCloud<pcl::PointXYZI>);
  for (auto ping : clouds_) {
    sensor_msgs::PointCloud2::Ptr transformed_ping(new sensor_msgs::PointCloud2);
    *transformed_ping = buffer.transform(*ping,frame);
    pcl::PointCloud<pcl::PointXYZI> pcl_cloud;
    pcl::fromROSMsg(*transformed_ping,pcl_cloud);
    out->getCloud()->operator+=(pcl_cloud);
  }
  out->computeCenterOfMass();
  return out;
}

}
