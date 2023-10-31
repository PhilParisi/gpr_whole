#include "abstract_mapper.h"

void AbstractMapper::Params::fromServer(ros::NodeHandlePtr nh){
    //nh->param<float>("map_res",map_res,0.4f);
  std::vector<std::string> default_topics{};
  nh->param<std::vector<std::string>>("cloud_topics", cloud_topics,
                                            default_topics);
  nh->param("queue_timeout",queue_timeout,  5.0f);

  nh->param("map/viz_period" ,map.viz_period,     10.0f);
  nh->param("map/write_period", map.write_period, 10.0f);
  nh->param<std::string>("map/filename", map.filename, "");
  nh->param<std::string>("map/frame_id" , map.frame_id, "map");
  fromServerImpl(nh);
}

AbstractMapper::AbstractMapper()
{
    nh_.reset(new ros::NodeHandle("~"));
//    params_.reset(new Params);
//    params_->fromServer(nh_);

//    initialize(nh_,params_);
}

void AbstractMapper::initialize(ros::NodeHandlePtr nh, AbstractMapper::Params::Ptr cfg){
    nh_=nh;
    params_=cfg;

    if(params_->map.filename!=""){
        save_timer_ = nh_->createTimer(ros::Duration(params_->map.write_period),&AbstractMapper::saveTimerCallback,this);
        ROS_INFO("%s is using map file %s ",ros::this_node::getName().c_str(), params_->map.filename.c_str());
    }

    if(params_->map.viz_period>0){
        viz_timer_ = nh_->createTimer(ros::Duration(params_->map.viz_period),&AbstractMapper::pubTimerCallback,this);
        ROS_INFO("%s is publishing map with period: %f ",ros::this_node::getName().c_str(),params_->map.viz_period);
    }else{
        ROS_INFO("%s viz_priod( %f ) is negative.   Not publishing visualization ",ros::this_node::getName().c_str(),params_->map.viz_period);
    }

    for (std::string topic : params_->cloud_topics) {
      cloud_subs_.push_back(nh_->subscribe<sensor_msgs::PointCloud2>(
          topic, 5000, &AbstractMapper::cloudCallback, this));
      ROS_INFO("%s is subscribing to %s ", ros::this_node::getName().c_str(),
               topic.c_str());
    }

    tf_listener_ = new tf2_ros::TransformListener(tf_buffer_);

    initMap();
}

void AbstractMapper::initMap(){
  if(params_->map.filename == "")
    resetMap();
  if(!loadMap())  // if there's no map to load reset it;
    resetMap();
}


void AbstractMapper::cloudCallback(
    const sensor_msgs::PointCloud2::ConstPtr &cloud) {
  cloud_queue_.push_back(cloud);
  processQueue();
  //updateMap();
  return;
}

void AbstractMapper::popTimeoutData(){
  std::string source_frame = cloud_queue_.front()->header.frame_id;
  std::string target_frame = params_->map.frame_id;

  if(tf_buffer_.canTransform(target_frame,source_frame,ros::Time(0))){  // check if the reference frames exist
    auto latest_tf = tf_buffer_.lookupTransform(target_frame,source_frame,ros::Time(0));

    ros::Duration timeout(params_->queue_timeout);

    if(cloud_queue_.size()>0 && latest_tf.header.stamp-cloud_queue_.front()->header.stamp > timeout){
      ROS_WARN("message waited too long for tf... throwing out");
      cloud_queue_.pop_front();
    }
  }
}

void AbstractMapper::processQueue(){

  while(cloud_queue_.size()>0){
    std::string source_frame = cloud_queue_.front()->header.frame_id;
    std::string target_frame = params_->map.frame_id;
    auto time = cloud_queue_.front()->header.stamp;
    if(tf_buffer_.canTransform(target_frame,source_frame,time)){
      processCloud(cloud_queue_.front());
      cloud_queue_.pop_front();
    }
    else {
      popTimeoutData();
    }

  }
}

