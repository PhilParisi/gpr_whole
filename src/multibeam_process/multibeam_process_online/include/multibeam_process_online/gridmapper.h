#ifndef GRIDMAPPER_H
#define GRIDMAPPER_H

#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include "tf2_ros/message_filter.h"
#include "tf2_ros/transform_listener.h"
#include <boost/filesystem.hpp>
#include <cmath>
#include <grid_map_msgs/GridMap.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <memory>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include "abstract_mapper.h"

class GridMapper: public AbstractMapper {
public:
  GridMapper();
  void resetMap();
  bool loadMap();
  void publish();
  //void spin();
  //void spinOnce();
  //void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr &cloud);
  //void vizTimerCallback(const ros::TimerEvent &);
  //void writeTimerCallback(const ros::TimerEvent &);
  void processCloud(const sensor_msgs::PointCloud2::ConstPtr &cloud);
  void saveMap();

  struct Params: public AbstractMapper::Params {
  public:
    struct {
      double res;
      double len_x;
      double len_y;
      double center_x;
      double center_y;
    } gridmap;
    typedef std::shared_ptr<Params> Ptr;
  protected:
    void fromServerImpl(ros::NodeHandlePtr node_ptr);
  };

protected:
  //ros::NodeHandlePtr node_;
  //std::deque<sensor_msgs::PointCloud2>
  //    cloud_queue_; ///<  Stores scans untill they we have a TF from the buffer
  //ros::Timer viz_timer_;
  ros::Timer write_timer_;
  //f2_ros::Buffer tf_buffer_;
  //tf2_ros::TransformListener *tf_listener_;
  //std::vector<ros::Subscriber> cloud_subs_;
  ros::Publisher gridmap_publisher_;

  std::shared_ptr<grid_map::GridMap> map_;
  Params::Ptr params_;
};

#endif // GRIDMAPPER_H
