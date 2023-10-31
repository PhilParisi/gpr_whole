#include "gridmapper.h"

void GridMapper::Params::fromServerImpl(ros::NodeHandlePtr node_ptr) {
  // map params
  node_ptr->param("map/res",                  gridmap.res,    1.0);
  node_ptr->param("map/len_x",                gridmap.len_x,  200.0);
  node_ptr->param("map/len_y",                gridmap.len_y,  -1.0);
  if (gridmap.len_y <= 0.0) // if no len_x is specified make len_x = len_y
    gridmap.len_y = gridmap.len_x;
  node_ptr->param("map/center_x",              gridmap.center_x, 0.0);
  node_ptr->param("map/center_y",              gridmap.center_y, 0.0);

}

GridMapper::GridMapper() {
  nh_.reset(new ros::NodeHandle("~"));
  params_.reset(new Params);
  params_->fromServer(nh_);
  initialize(nh_,params_);

  gridmap_publisher_ =
      nh_->advertise<grid_map_msgs::GridMap>("grid_map", 1, true);
  ROS_INFO("starting...");
}

void GridMapper::resetMap() {
  map_.reset(new grid_map::GridMap({"elevation", "N", "SUM"}));
  map_->setFrameId(params_->map.frame_id);
  grid_map::Position map_center;
  map_center.x() = params_->gridmap.center_x;
  map_center.y() = params_->gridmap.center_y;
  map_->setGeometry(grid_map::Length(params_->gridmap.len_x, params_->gridmap.len_y),
                    params_->gridmap.res, map_center);
  ROS_INFO("Created map with size %f x %f m (%i x %i cells).",
           map_->getLength().x(), map_->getLength().y(), map_->getSize()(0),
           map_->getSize()(1));
  map_->setFrameId(params_->map.frame_id);

}

void GridMapper::publish() {
  ROS_INFO("Gridmap Publishing");
  ros::Time time = ros::Time::now();
  map_->setTimestamp(time.toNSec());
  grid_map_msgs::GridMap msg;
  grid_map::GridMapRosConverter::toMessage(*map_, msg);
  gridmap_publisher_.publish(msg);
  return;
}

void GridMapper::saveMap() {
  if (grid_map::GridMapRosConverter::saveToBag(*map_, params_->map.filename,
                                               "saved_grid"))
    ROS_INFO("grid saved");
  else
    ROS_ERROR("grid failed to save at %s", params_->map.filename.c_str());
}

bool GridMapper::loadMap() {
  if (boost::filesystem::exists(params_->map.filename)) {
    ROS_INFO("Grid Map found on disk, loading...");
    map_.reset(new grid_map::GridMap({"elevation", "N", "SUM"}));
    grid_map::GridMapRosConverter::loadFromBag(params_->map.filename,
                                               "saved_grid", *map_);
    return true;
  } else {
    ROS_INFO("No Grid Map found at %s.  Creating a new one...",
             params_->map.filename.c_str());
    return false;
  }
}

void GridMapper::processCloud(const sensor_msgs::PointCloud2::ConstPtr &cloud) {
  sensor_msgs::PointCloud2 mapframe_cloud;
  pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
  mapframe_cloud =
      tf_buffer_.transform(*cloud, params_->map.frame_id);
  pcl::fromROSMsg(mapframe_cloud, pcl_cloud);

  for (size_t i = 0; i < pcl_cloud.points.size(); ++i) {
    grid_map::Position position;
    position.x() = double(pcl_cloud.points[i].x);
    position.y() = double(pcl_cloud.points[i].y);

    try {
      if (std::isnan(map_->atPosition("N", position))) {
        map_->atPosition("N", position) = 0;
      };
      if (std::isnan(map_->atPosition("SUM", position))) {
        map_->atPosition("SUM", position) = 0;
      };
      if (std::isnan(map_->atPosition("elevation", position))) {
        map_->atPosition("elevation", position) = 0;
      };

      map_->atPosition("N", position) = map_->atPosition("N", position) + 1;
      map_->atPosition("SUM", position) =
          map_->atPosition("SUM", position) + pcl_cloud.points[i].z;
      map_->atPosition("elevation", position) =
          map_->atPosition("SUM", position) /
          map_->atPosition("N", position);
    } catch (std::out_of_range) {
      ROS_INFO("point (%f,%f) is outside the gridmap",
               pcl_cloud.points[i].x, pcl_cloud.points[i].y);
    }
  }


}
