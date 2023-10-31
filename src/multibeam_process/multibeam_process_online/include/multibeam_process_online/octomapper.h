#ifndef OCTOMAPPER_H
#define OCTOMAPPER_H

#include "abstract_mapper.h"

#include <octomap/octomap.h>
#include <octomap/OcTree.h>
#include <visualization_msgs/MarkerArray.h>


class Octomapper: public AbstractMapper
{
public:
  Octomapper();
  void resetMap();
  bool loadMap();
  void publish();
  void processCloud(const sensor_msgs::PointCloud2::ConstPtr &cloud_msg);
  void saveMap();
  struct Params: public AbstractMapper::Params {
  public:
    struct {
      double res;
      std::string output_pointcloud;
    } octomap;
    typedef std::shared_ptr<Params> Ptr;
  protected:
    void fromServerImpl(ros::NodeHandlePtr node_ptr);
  };

protected:
  Params::Ptr params_;
  std::shared_ptr<octomap::OcTree> mapPtr_;  ///< A pointer to the map.  Represented as an octomap::OcTree
  ros::Publisher marker_pub_;
  ros::Publisher point_cloud_pub_;
};

#endif // OCTOMAPPER_H
