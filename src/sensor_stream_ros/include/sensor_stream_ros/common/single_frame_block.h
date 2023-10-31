#ifndef SINGLE_FRAME_BLOCK_H
#define SINGLE_FRAME_BLOCK_H

#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/random_sample.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include "abstract_block.h"

namespace ss {

/*!
 * \brief The Block class represents a block of points that you want to add to a GPR. It has both a
 * pcl point cloud and a CudaMat<float> representation.
 */
class SingleFrameBlock : public AbstractBlock{
public:
  typedef std::shared_ptr<SingleFrameBlock> Ptr;
    SingleFrameBlock();
    SingleFrameBlock(BlockParams::Ptr params);
    SingleFrameBlock(BlockParams::Ptr params,pcl::PointCloud<pcl::PointXYZI>::Ptr cloud);
    void setPointcloud(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud);
    void push_back(pcl::PointXYZI pt);
    void computeCenterOfMass();
    pcl::PointXYZ getCenterOfMass();
    size_t size(){return _cloud->size();}
    pcl::PointCloud<pcl::PointXYZI>::Ptr getCloud(){return _cloud;}
    void clear();

    TrainingBlock getTrainingData();
    TrainingBlock getSubsampledTrainingData(size_t block_size);


protected:
    pcl::PointXYZ centerOfMass;
    //std::shared_ptr<BlockParams> _params;
    pcl::PointCloud<pcl::PointXYZI>::Ptr _cloud;

};

}



#endif // SINGLE_FRAME_BLOCK_H
