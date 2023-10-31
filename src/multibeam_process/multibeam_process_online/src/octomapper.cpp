#include "octomapper.h"

void Octomapper::Params::fromServerImpl(ros::NodeHandlePtr node_ptr){
  node_ptr->param("map/res",                  octomap.res,    1.0);
  node_ptr->param<std::string>("map/output_pointcloud",octomap.output_pointcloud,"");
}

Octomapper::Octomapper()
{
  nh_.reset(new ros::NodeHandle("~"));
  params_.reset(new Params);
  params_->fromServer(nh_);
  initialize(nh_,params_);

  marker_pub_ = nh_->advertise<visualization_msgs::MarkerArray>("occupied_cells_vis_array",1 ,true);
  point_cloud_pub_ =  nh_->advertise<sensor_msgs::PointCloud2>("octomap_point_cloud_centers",1,true);

}

void Octomapper::resetMap(){
  mapPtr_=std::make_shared<octomap::OcTree>(params_->octomap.res);
}

bool Octomapper::loadMap(){
  mapPtr_=std::make_shared<octomap::OcTree>(params_->octomap.res);
  if(mapPtr_->readBinary(params_->map.filename)){
      ROS_INFO("Octomap found on disk, loading...");
      return true;
  }else {
      ROS_INFO("No Octomap found at map_filename location.  Creating a new one...");
      return false;
  }
}

std_msgs::ColorRGBA heightMapColor(double h){
  std_msgs::ColorRGBA color;
    color.a = 1.0;
    // blend over HSV-values (more colors)

    double s = 1.0;
    double v = 1.0;

    h -= floor(h);
    h *= 6;
    int i;
    double m, n, f;

    i = floor(h);
    f = h - i;
    if (!(i & 1))
      f = 1 - f; // if i is even
    m = v * (1 - s);
    n = v * (1 - s * f);

    switch (i) {
      case 6:
      case 0:
        color.r = v; color.g = n; color.b = m;
        break;
      case 1:
        color.r = n; color.g = v; color.b = m;
        break;
      case 2:
        color.r = m; color.g = v; color.b = n;
        break;
      case 3:
        color.r = m; color.g = n; color.b = v;
        break;
      case 4:
        color.r = n; color.g = m; color.b = v;
        break;
      case 5:
        color.r = v; color.g = m; color.b = n;
        break;
      default:
        color.r = 1; color.g = 0.5; color.b = 0.5;
        break;
  }

    return color;
}

void Octomapper::publish(){
  ROS_INFO("Octomap Publishing");
  size_t octomapSize = mapPtr_->size();
  // TODO: estimate num occ. voxels for size of arrays (reserve)
  if (octomapSize <= 1){
    ROS_WARN("%s: Nothing to publish, octree is empty", ros::this_node::getName().c_str());
    return;
  }
  // init markers:
  visualization_msgs::MarkerArray occupiedNodesVis;
  // each array stores all cubes of a different size, one for each depth level:
  unsigned m_treeDepth = mapPtr_->getTreeDepth();
  occupiedNodesVis.markers.resize(m_treeDepth+1);

  pcl::PointCloud<pcl::PointXYZ> cloud;

  for (octomap::OcTree::iterator it = mapPtr_->begin(),end = mapPtr_->end(); it != end; ++it){
    if (mapPtr_->isNodeOccupied(*it)){
      double x = it.getX();
      double y = it.getY();
      double z = it.getZ();
      double size = it.getSize();

      cloud.push_back(pcl::PointXYZ(x, y, z));

      // add to marker array
      unsigned idx = it.getDepth();
      assert(idx < occupiedNodesVis.markers.size());

      geometry_msgs::Point cubeCenter;
      cubeCenter.x = x;
      cubeCenter.y = y;
      cubeCenter.z = z;

      occupiedNodesVis.markers[idx].points.push_back(cubeCenter);
//      if (m_useHeightMap){
        double minX, minY, minZ, maxX, maxY, maxZ;
        mapPtr_->getMetricMin(minX, minY, minZ);
        mapPtr_->getMetricMax(maxX, maxY, maxZ);

        double h = (1.0 - std::min(std::max((cubeCenter.z-minZ)/ (maxZ - minZ), 0.0), 1.0)) *0.8;
        occupiedNodesVis.markers[idx].colors.push_back(heightMapColor(h));
//      }
    }
  }

  sensor_msgs::PointCloud2 cloudMsg;
  pcl::toROSMsg (cloud, cloudMsg);
  cloudMsg.header.frame_id = params_->map.frame_id;
  cloudMsg.header.stamp = ros::Time::now();
  point_cloud_pub_.publish(cloudMsg);


  for (unsigned i= 0; i < occupiedNodesVis.markers.size(); ++i){
        double size = mapPtr_->getNodeSize(i);

        occupiedNodesVis.markers[i].header.frame_id = params_->map.frame_id;
        occupiedNodesVis.markers[i].header.stamp = ros::Time::now();
        occupiedNodesVis.markers[i].ns = "map";
        occupiedNodesVis.markers[i].id = i;
        occupiedNodesVis.markers[i].type = visualization_msgs::Marker::CUBE_LIST;
        occupiedNodesVis.markers[i].scale.x = size;
        occupiedNodesVis.markers[i].scale.y = size;
        occupiedNodesVis.markers[i].scale.z = size;
//        if (!m_useColoredMap)
//          occupiedNodesVis.markers[i].color = m_color;


        if (occupiedNodesVis.markers[i].points.size() > 0)
          occupiedNodesVis.markers[i].action = visualization_msgs::Marker::ADD;
        else
          occupiedNodesVis.markers[i].action = visualization_msgs::Marker::DELETE;
      }

  marker_pub_.publish(occupiedNodesVis);
}
void Octomapper::processCloud(const sensor_msgs::PointCloud2::ConstPtr &cloud_msg){

  // declare vars
  sensor_msgs::PointCloud2 cloud;
  sensor_msgs::PointCloud2 inCloud;
  octomap::Pointcloud octoCloud;
  pcl::PointCloud<pcl::PointXYZ> pcl_cloud;

  // convert point cloud to octmap
  //projector_.transformLaserScanToPointCloud("map",_scanQueue.front(), cloud, _tfBuffer);
  cloud = tf_buffer_.transform(*cloud_msg,params_->map.frame_id);
  pcl::fromROSMsg (cloud, pcl_cloud);
  for (pcl::PointCloud<pcl::PointXYZ>::const_iterator it = pcl_cloud.begin(); it != pcl_cloud.end(); ++it){
    octoCloud.push_back(it->x, it->y, it->z);
    //ROS_INFO("adding points");
  }

  // prepare sensor origin
  geometry_msgs::PointStamped sensorOrigin,sensorOriginMap;
  sensorOrigin.header=cloud_msg->header;
  sensorOrigin.point.x=0;
  sensorOrigin.point.y=0;
  sensorOrigin.point.z=0;
  sensorOriginMap=tf_buffer_.transform(sensorOrigin,params_->map.frame_id);
  octomap::point3d sensorOriginOcto(sensorOriginMap.point.x,sensorOriginMap.point.y,sensorOriginMap.point.z);

  // insert into octomap
  mapPtr_->insertPointCloud(octoCloud,sensorOriginOcto);

}
void Octomapper::saveMap(){
  ROS_INFO("writing octomap to %s", params_->map.filename.c_str());
  if(!mapPtr_->writeBinary(params_->map.filename)){
      ROS_ERROR("Error writing map to %s", params_->map.filename.c_str());
  }
  pcl::PointCloud<pcl::PointXYZ> cloud;

}
