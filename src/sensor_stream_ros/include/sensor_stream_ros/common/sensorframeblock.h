#ifndef SENSORFRAMEBLOCK_H
#define SENSORFRAMEBLOCK_H

#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <deque>
#include <tf2_ros/buffer.h>
#include <exception>
#include <mutex>
#include <sensor_stream/include/sensor_stream/cudablockmat.h>
#include "abstract_block.h"
#include <sensor_stream_ros/common/single_frame_block.h>

namespace ss{


/*!
 * \brief Represents a sensor stream cudamat of points using a pcl cloud
 */
class SensorFrameBlock:  public AbstractBlock
{
public:
  typedef std::shared_ptr<SensorFrameBlock> Ptr;
  SensorFrameBlock(BlockParams::Ptr params);

  /*!
   * \brief does the size of the cloud == the BlockParams::size set using setParams;
   * \return this->size == BlockParams::size
   */
  bool isFull();

  /*!
   * \brief adds a single point to the ping at the back of the queue
   * \param point the point you want to add
   */
  void addPoint(pcl::PointXYZI point);

  /*!
   * \brief adds an empty ping to the cloud_ queue.   This is the ping that addPoint will work on
   * \throws std::runtime_error if too many points were added to the block
   */
  void addPing(std_msgs::Header header);

  /*!
   * \brief Determines the number of points in the cloud
   * \return the number of points in the cloud
   */
  size_t size();

  /*!
   * \brief transform the set of pings to the desired frame
   * \param buffer the tf2_ros::Buffer you want to use to make the transformation
   * \param frame the frame you want to change to
   * \return
   */
  SensorFrameBlock::Ptr transform(tf2_ros::Buffer & buffer, std::string frame);

  /*!
   * \brief transforms the block to a static frame and returns a SingleFrameBlock
   * \param buffer tf2_ros::Buffer you want to use to make the transformation
   * \param frame the STATIC frame you want to change to
   * \return
   */
  SingleFrameBlock::Ptr transform2StaticFrame(tf2_ros::Buffer & buffer, std::string frame);

  /*!
   * \brief returns a reference to the individual cloud messages in the cloud array
   * \return  a reference to the dequeue of clouds in the block
   */
  std::deque<sensor_msgs::PointCloud2::Ptr> & getClouds(){return clouds_;}

  size_t swath;

protected:
  void clearBuffer();
  std::mutex read_mtx_;
  std::deque<sensor_msgs::PointCloud2::Ptr> clouds_;  ///< \brief the clouds in sensor frame that represent hold the points in the block
  std_msgs::Header buffer_header_;
  pcl::PointCloud<pcl::PointXYZI>::Ptr buffer_cloud_;
  size_t size_;

};

}

#endif // SENSORFRAMEBLOCK_H
