#ifndef ABSTRACT_MAPPER_H
#define ABSTRACT_MAPPER_H

#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/message_filter.h>
#include <tf2_ros/transform_listener.h>
#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <memory>



class AbstractMapper
{
public:
  struct Params{
  public:
    std::vector<std::string> cloud_topics;
    float queue_timeout;
    struct {
      std::string filename;  ///< filename to save the map to so that it can be reloaded
      std::string frame_id;
      float write_period;
      float viz_period;
    }map;
      void fromServer(ros::NodeHandlePtr nh); ///< get params from ros param server
      typedef std::shared_ptr<Params> Ptr;
  protected:
      virtual void fromServerImpl(ros::NodeHandlePtr nh){return;}
  };

  AbstractMapper();
  void initialize(ros::NodeHandlePtr nh,Params::Ptr cfg);
  virtual void initMap();
  virtual bool loadMap(){ROS_WARN("loadMap() is not implemented for %s", ros::this_node::getName().c_str()); return false;}
  virtual void resetMap(){ROS_WARN("resetMap() is not implemented for %s", ros::this_node::getName().c_str());}
  virtual void saveMap(){ROS_WARN("saveMap() is not implemented for %s", ros::this_node::getName().c_str());}
  void saveTimerCallback(const ros::TimerEvent&){saveMap();}
  virtual void publish(){ROS_WARN("publish() is not implemented for %s", ros::this_node::getName().c_str());}
  void pubTimerCallback(const ros::TimerEvent&){publish();}
  //virtual void updateMap(){ROS_WARN("updateMap() is not implemented for %s", ros::this_node::getName().c_str());}
  virtual void processCloud(const sensor_msgs::PointCloud2::ConstPtr &cloud){ROS_WARN("processCloud() is not implemented for %s", ros::this_node::getName().c_str());}
  void cloudCallback( const sensor_msgs::PointCloud2::ConstPtr &cloud);
  void processQueue();

  void spin(){ros::spin();}
  void spinOnce(){ros::spinOnce();}

protected:
  void popTimeoutData();

  ros::NodeHandlePtr    nh_;            ///< a pointer to the ros node handle
  ros::Timer            save_timer_;     ///< a timer to save the results periodically
  ros::Timer            viz_timer_;      ///< a timer to publish the map
  Params::Ptr           params_;           ///< a shared pointer to the config
  tf2_ros::Buffer       tf_buffer_;
  tf2_ros::TransformListener *tf_listener_;
  std::deque<sensor_msgs::PointCloud2::ConstPtr>
                        cloud_queue_;
  std::vector<ros::Subscriber> cloud_subs_;
};

#endif // ABSTRACT_MAPPER_H
