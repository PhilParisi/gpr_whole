#ifndef BLOCKARRAY_H
#define BLOCKARRAY_H
#include "sensorframeblock.h"

#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

namespace ss{

struct BlockArrayParams{
  BlockArrayParams(){
    divisions = 4;
  }
  size_t divisions;
  BlockParams::Ptr block_params;
};

class BlockArray:public std::list<SensorFrameBlock::Ptr>
{
public:
  BlockArray();
  void addPing(sensor_msgs::PointCloud2::ConstPtr ping);
  void addEmptyBlock();
  BlockArrayParams params;
  //std::deque<SensorFrameBlock::Ptr>::iterator end(){return blocks_.end();}
  //std::deque<SensorFrameBlock::Ptr>::iterator begin(){return blocks_.begin();}
protected:
  //std::deque<SensorFrameBlock::Ptr> blocks_;
  std::vector<SensorFrameBlock::Ptr> partial_;
  //BlockParams::Ptr block_params_;
};

}

#endif // BLOCKARRAY_H
