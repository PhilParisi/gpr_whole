#ifndef PATCHTESTER_H
#define PATCHTESTER_H

#define PCL_NO_PRECOMPILE

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <geometry_msgs/TransformStamped.h>
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/message_filter.h"
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <urdf/model.h>
#include <tf2_msgs/TFMessage.h>
#include <tf2/buffer_core.h>
#include <pcl/visualization/cloud_viewer.h>
#include <iostream>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <atomic>
#include <thread>
#include <future>
#include <geometry_msgs/PointStamped.h>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <multibeam_process_core/tf_projector.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>

#include <pcl/pcl_macros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

struct PTPoint
{
  PCL_ADD_POINT4D;                  // preferred way of adding a XYZ+padding
  float intensity;
  float ping_no;
  //PCL_MAKE_ALIGNED_OPERATOR_NEW     // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PTPoint,           // here we assume a XYZ + "test" (as fields)
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (float, intensity, intensity)
                                   (float,ping_no,ping_no)
)



class PatchTester
{
public:
  PatchTester();
  ~PatchTester();
  void openBag(std::string filename);
  void readBag(std::string pointcloudTopic);
  void readRobotModel(std::string filename);
  void savePCD(std::string filename);
  void setMapFrame(std::string mapFrame){mapFrame_=mapFrame;}
  std::string getMapFrame(){return mapFrame_;}
  pcl::PointCloud<PTPoint>::Ptr projectPoints();
  pcl::PointCloud<PTPoint>::Ptr projectPointsRange(int min, int max);
  pcl::PointCloud<PTPoint>::Ptr projectPointsThreaded(int min, int max);
  pcl::PointCloud<PTPoint>::Ptr projectPointsThreaded();
  sensor_msgs::PointCloud2::Ptr projectPoints(size_t i);
  sensor_msgs::PointCloud2::Ptr getProjectedMsg();
  sensor_msgs::PointCloud2::Ptr getProjectedMsg(int min, int max);
  int getProgress(){return int(progress*100.0f);}
  int projectionFailures(){return projectionFailures_;}
  urdf::Model & getRobotModel(){return robotModel_;}
  tf2_ros::Buffer & getBuffer(){return tfBuffer_;}
  sensor_msgs::PointCloud2::ConstPtr getPing(size_t i){return detections_[i];}
  std::vector<sensor_msgs::PointCloud2::ConstPtr> getPingVect(){return detections_;}
  void updateStaticTf();
  void addStaticFrame(std::string parent, std::string child,
                      double x, double y, double z,
                      double yaw, double pitch, double roll);
  pcl::PointCloud<pcl::PointXYZI>::Ptr getFrameTrack(std::string frame);
  void saveFrameTrack(std::string frame, std::string filename);

  std::vector<rosbag::ConnectionInfo> pointcloudConnections;
  std::vector<std::string> frame_ids;
protected:
  void addStaticTf(urdf::LinkConstSharedPtr link);
  rosbag::Bag bag_;
  std::vector<sensor_msgs::PointCloud2::ConstPtr> detections_;
  tf2_ros::Buffer tfBuffer_;
  urdf::Model robotModel_;
  ros::Time startTime_;
  pcl::PointCloud<PTPoint>::Ptr outputCloud;
  std::atomic<float> progress;
  std::atomic_int projectionFailures_;
  std::string mapFrame_;

};


#endif // PATCHTESTER_H
