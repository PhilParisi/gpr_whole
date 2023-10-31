#ifndef GPR_MAPPER_H
#define GPR_MAPPER_H
#include <sensor_stream/blockgpr.h>
#include <sensor_stream/cudamat.h>
#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include "ros/time.h"
#include "geometry_msgs/PointStamped.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/message_filter.h"
#include "message_filters/subscriber.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/point_cloud2_iterator.h"
#include "sensor_msgs/point_cloud_conversion.h"
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <queue>
#include "block_tiler.h"
#include "ss_cv.h"
#include <nvToolsExt.h>
#include "yaml-cpp/yaml.h"
#include <thread>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_stream/include/sensor_stream/gpr/sqexpsparse2d.h>

struct GprMapperConfig{
    std::string inCloudTopic;
    std::string mapFrame;
    int swathDivisions;
    float vizRate;
    float max_chol_size;
    ss::ros::TileParamsPtr tile;
    ss::BlockParams::Ptr block;
    gpr::GprParams gpr;
    void getFromServer(ros::NodeHandlePtr node);
    void writeToYaml(std::string filename);
};

class GprMapper
{
public:
  GprMapper();

  GprMapperConfig cfg;
  void inCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& inCloud);
  void addToBlock(pcl::PointXYZI pt, size_t swathNum);
  void addBlockToGpr(size_t swathNum);
  void predict(float xCenter, float yCenter);
  void doTile(ss::ros::Tile workingTile);
  void spinOnce();
  void spin();
  void vizTimerCallback(const ros::TimerEvent&);
  tf2_ros::Buffer & tfBuffer();
  tf2_ros::Buffer _tfBuffer;
  bool tileQueueHasElements(){return _tileQueue.size()>0;}
  ss::ros::Tile queueFront(){return _tileQueue.front();}
  std::shared_ptr<BlockGpr> gpr;
  std::deque<ss::ros::Tile> & getQueue(){return _tileQueue;}
  void publishPredictions();
  void publishSparsity();
private:
  //BlockGpr _gpr;
  ss::ros::BlockTiler _tiler;
  ros::NodeHandlePtr _node;
  std::shared_ptr<image_transport::ImageTransport> _imTrans;
  image_transport::Publisher _sparsityPub;
  ros::Subscriber _inCloudSub;

  tf2_ros::TransformListener * _tfListener;
  std::vector<ss::SingleFrameBlock> _inputBlocks;
  pcl::PointCloud<pcl::PointXYZI> _blockCloud;
  ros::Publisher _blockCloudPub;
  ros::Publisher _predCloudPub;
  size_t _blockCount;
  std::deque<ss::ros::Tile> _tileQueue;
  ros::Timer _vizTimer;
  float _vizRate;
};

#endif // GPR_MAPPER_H
