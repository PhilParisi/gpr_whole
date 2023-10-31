#include "patchtester.h"

PatchTester::PatchTester():tfBuffer_(ros::Duration(1000000000,0))
{
  tfBuffer_.setUsingDedicatedThread(true);
  progress = 0.0f;
  projectionFailures_ = 0;
}

PatchTester::~PatchTester(){
  bag_.close();
}

void PatchTester::openBag(std::string filename){
  std::cout << "opening bag" << std::endl;
  bag_.open(filename);
  std::cout << "searching for topics" << std::endl;
  rosbag::View topic_view(bag_);
  for (const rosbag::ConnectionInfo* info: topic_view.getConnections()) {
    if(info->datatype=="sensor_msgs/PointCloud2"){
      pointcloudConnections.push_back(*info);
    }
  }

}

void PatchTester::readBag(std::string pointcloudTopic){

  std::vector<std::string> topics;
  topics.push_back(std::string(pointcloudTopic));
  topics.push_back(std::string("/tf"));
  bool first = true;

  for(rosbag::MessageInstance const m: rosbag::View(bag_,rosbag::TopicQuery(topics)))
  {
    if(first)
      startTime_=m.getTime();

    sensor_msgs::PointCloud2::ConstPtr detection = m.instantiate<sensor_msgs::PointCloud2>();
    if (detection){
      detections_.push_back(detection);
    }

    tf2_msgs::TFMessage::ConstPtr transform = m.instantiate<tf2_msgs::TFMessage>();
    if(transform){
      for(size_t i =0 ; i<transform->transforms.size();i++){
        tfBuffer_.setTransform(transform->transforms[i],"patch_tester");
      }
    }
  }

  std::cout << "number of pings found: " << detections_.size() << std::endl;

  YAML::Node frames = YAML::Load(tfBuffer_.allFramesAsYAML());

  std::cout << tfBuffer_.allFramesAsYAML() << std::endl;
  for(YAML::const_iterator it=frames.begin();it!=frames.end();++it) {
    std::cout << "frame: " << it->first.as<std::string>() << "   parent: " << it->second["parent"].as<std::string>() << "\n";
    if (std::find(frame_ids.begin(), frame_ids.end(), it->first.as<std::string>()) == frame_ids.end())
            frame_ids.push_back(  it->first.as<std::string>()  );
    if (std::find(frame_ids.begin(), frame_ids.end(), it->second["parent"].as<std::string>()) == frame_ids.end())
            frame_ids.push_back(  it->second["parent"].as<std::string>()  );
  }
  //std::cout << tfBuffer_.allFramesAsString() << std::endl;
}

void PatchTester::readRobotModel(std::string filename){
  if (!robotModel_.initFile(filename)){
    ROS_ERROR("Failed to parse urdf file");
    return;
  }
  updateStaticTf();
}

pcl::PointCloud<PTPoint>::Ptr PatchTester::projectPoints(){
  std::cout << "projecting" << std::endl;
  //addStaticTf();
  outputCloud.reset(new pcl::PointCloud<PTPoint>);
//  projectionFailures_ = 0;
//  for(size_t i = 0; i<detections_.size() ; i++){
//    try{
//      try{
//        pcl::PointCloud<pcl::PointXYZI> transformedCloud;
//        sensor_msgs::PointCloud2 transformedMsg= tfBuffer_.transform(*detections_[i],mapFrame_);
//        pcl::fromROSMsg(transformedMsg,transformedCloud);

//        *outputCloud += transformedCloud;
//      }catch(tf2::LookupException& e){
//        projectionFailures_++;
//      }
//    }catch(tf2::ExtrapolationException& e){
//      projectionFailures_++;
//    }
//    progress = float(i)/float(detections_.size());
//  }
//  progress = 1.0f;
  return outputCloud;
}

pcl::PointCloud<PTPoint>::Ptr PatchTester::projectPointsRange(int min, int max){
  pcl::PointCloud<PTPoint>::Ptr out;
  out.reset(new pcl::PointCloud<PTPoint>);
  projectionFailures_ = 0;
  if (max>detections_.size()-1){
      max = detections_.size()-1;
  }
  if (min < 0){
      min = 0;
  }
  for(size_t i = min; i<=max ; i++){
    try{
      try{
        pcl::PointCloud<pcl::PointXYZI> transformedCloud;
        sensor_msgs::PointCloud2 transformedMsg= tfBuffer_.transform(*detections_[i],mapFrame_);
        pcl::fromROSMsg(transformedMsg,transformedCloud);
        pcl::PointCloud<PTPoint> subcloud;
        subcloud.resize(transformedCloud.size());

        for (size_t j = 0 ; j<transformedCloud.size(); j++){
          //it->intensity = i;
            subcloud[j].x = transformedCloud[j].x;
            subcloud[j].y = transformedCloud[j].y;
            subcloud[j].z = transformedCloud[j].z;
            subcloud[j].intensity = transformedCloud[j].intensity;
            subcloud[j].ping_no = i;
        }

        *out += subcloud;

        progress = progress + 1.0f/float(detections_.size());
      }catch(tf2::LookupException& e){
        projectionFailures_++;
      }
    }catch(tf2::ExtrapolationException& e){
      projectionFailures_++;
    }
  }
  return out;
}

pcl::PointCloud<PTPoint>::Ptr PatchTester::projectPointsThreaded(int min, int max){
    updateStaticTf();
    progress = 0;
    projectionFailures_ = 0;
    if (max>detections_.size()-1){
        max = detections_.size()-1;
    }
    if (min < 0){
        min = 0;
    }
    size_t threadCount = std::thread::hardware_concurrency();
    pcl::PointCloud<PTPoint>::Ptr out(new pcl::PointCloud<PTPoint>);
    std::vector<std::future<pcl::PointCloud<PTPoint>::Ptr > > values;
    std::vector<std::thread> threads;
    for(size_t i = 0 ; i <threadCount ;i++ ){
        int range = max - min;
        int div = range/threadCount;
        int thrMin = min+div*i;
        int thrMax = min+div*i+div;
        if(i==threadCount-1){
            thrMax = max;
        }
        values.push_back( std::async(&PatchTester::projectPointsRange,this,thrMin,thrMax) );
    }
    for(size_t i = 0 ; i <threadCount ;i++ ){
        *out+=*values[i].get();
    }
    progress = 1;
    return out;
}

pcl::PointCloud<PTPoint>::Ptr PatchTester::projectPointsThreaded(){

    outputCloud=projectPointsThreaded(0,detections_.size()-1);
    return  outputCloud;
}

sensor_msgs::PointCloud2::Ptr PatchTester::projectPoints(size_t i){
    //addStaticTf();
    sensor_msgs::PointCloud2::Ptr transformedMsg(new sensor_msgs::PointCloud2);
    try{
      *transformedMsg = tfBuffer_.transform(*detections_[i],mapFrame_);
    }catch(tf2::LookupException& e){

    }
    return transformedMsg;
}

sensor_msgs::PointCloud2::Ptr PatchTester::getProjectedMsg(){
    sensor_msgs::PointCloud2::Ptr cloudMsg(new sensor_msgs::PointCloud2);
    pcl::toROSMsg(*outputCloud, *cloudMsg);
    cloudMsg->header.stamp=ros::Time::now();
    cloudMsg->header.frame_id = mapFrame_;
    return cloudMsg;
}

sensor_msgs::PointCloud2::Ptr PatchTester::getProjectedMsg(int min, int max){
    if(min==max){
        sensor_msgs::PointCloud2::Ptr emptyMsg(new sensor_msgs::PointCloud2);
        return emptyMsg;
    }
    pcl::PointCloud<PTPoint>::Ptr pclCloud = projectPointsThreaded(min,max);
    sensor_msgs::PointCloud2::Ptr cloudMsg(new sensor_msgs::PointCloud2);
    pcl::toROSMsg(*pclCloud, *cloudMsg);
    cloudMsg->header.stamp=ros::Time::now();
    cloudMsg->header.frame_id = mapFrame_;
    return cloudMsg;
}

void PatchTester::savePCD(std::string filename){
  std::cout << "saving" << std::endl;
  pcl::io::savePCDFileBinary(filename,*outputCloud);
}

void PatchTester::updateStaticTf(){
    urdf::LinkConstSharedPtr link = robotModel_.getRoot();
    addStaticTf(link);
}

void PatchTester::addStaticTf(urdf::LinkConstSharedPtr link){
  std::vector<urdf::LinkSharedPtr> children = link->child_links;
  std::vector<urdf::JointSharedPtr> joints = link->child_joints;
  for(size_t i = 0 ; i <joints.size(); i++){
    if(joints[i]->type==urdf::Joint::FIXED){
      //std::cout << joints[i]->name << ": "<< joints[i]->parent_to_joint_origin_transform.position.x <<" "<<joints[i]->parent_to_joint_origin_transform.position.y <<" "<<joints[i]->parent_to_joint_origin_transform.position.z << std::endl;
      urdf::Pose pose = joints[i]->parent_to_joint_origin_transform;
      geometry_msgs::TransformStamped transform;
      transform.header.stamp = startTime_;
      transform.header.frame_id = joints[i]->parent_link_name;
      transform.child_frame_id  = joints[i]->child_link_name;
      transform.transform.translation.x = pose.position.x;
      transform.transform.translation.y = pose.position.y;
      transform.transform.translation.z = pose.position.z;
      transform.transform.rotation.x = pose.rotation.x;
      transform.transform.rotation.y = pose.rotation.y;
      transform.transform.rotation.z = pose.rotation.z;
      transform.transform.rotation.w = pose.rotation.w;
      tfBuffer_.setTransform(transform,"patch_tester",true);
    }
    else{
      std::cout << "warning: PatchTester only supports fixed urdf joints" << std::endl;
    }
  }
  for(size_t i = 0 ; i <children.size(); i++){
    addStaticTf(children[i]);
  }
}

void PatchTester::addStaticFrame(std::string parent, std::string child,
                                 double x, double y, double z,
                                 double yaw, double pitch, double roll){
    geometry_msgs::TransformStamped transform;
    tf2::Quaternion myQuaternion;
    myQuaternion.setRPY( roll, pitch, yaw );
    transform.header.stamp = startTime_;
    transform.header.frame_id = parent;
    transform.child_frame_id  = child;
    transform.transform.translation.x = x;
    transform.transform.translation.y = y;
    transform.transform.translation.z = z;
    transform.transform.rotation.x = myQuaternion.x();
    transform.transform.rotation.y = myQuaternion.y();
    transform.transform.rotation.z = myQuaternion.z();
    transform.transform.rotation.w = myQuaternion.w();
    tfBuffer_.setTransform(transform,"patch_tester",true);
}

pcl::PointCloud<pcl::PointXYZI>::Ptr PatchTester::getFrameTrack(std::string frame){
    pcl::PointCloud<pcl::PointXYZI>::Ptr outCloud(new pcl::PointCloud<pcl::PointXYZI>);
    for (size_t i =0 ; i< detections_.size(); i++) {
        try{
            geometry_msgs::PointStamped sensorOrigin,sensorOriginMap;
            sensorOrigin.header.frame_id=frame;
            sensorOrigin.header.stamp=detections_[i]->header.stamp;
            sensorOrigin.point.x=0;
            sensorOrigin.point.y=0;
            sensorOrigin.point.z=0;
            sensorOriginMap=tfBuffer_.transform(sensorOrigin,mapFrame_);
            pcl::PointXYZI framePoint;
            framePoint.x=sensorOriginMap.point.x;
            framePoint.y=sensorOriginMap.point.y;
            framePoint.z=sensorOriginMap.point.z;
            framePoint.intensity=detections_[i]->header.stamp.toSec();
            outCloud->push_back(framePoint);
        }catch(...){

        }
    }
    return outCloud;
}

void PatchTester::saveFrameTrack(std::string frame, std::string filename){

     std::ofstream myfile;
     myfile.open (filename);
     myfile << "Frame: " << frame << "\n";
     myfile << "x,y,z,unixtime\n";

     for (size_t i =0 ; i< detections_.size(); i++) {
         try{
             geometry_msgs::PointStamped sensorOrigin,sensorOriginMap;
             sensorOrigin.header.frame_id=frame;
             sensorOrigin.header.stamp=detections_[i]->header.stamp;
             sensorOrigin.point.x=0;
             sensorOrigin.point.y=0;
             sensorOrigin.point.z=0;
             sensorOriginMap=tfBuffer_.transform(sensorOrigin,mapFrame_);

             myfile << std::setprecision(17)
                    << sensorOriginMap.point.x << ","
                    << sensorOriginMap.point.y << ","
                    << sensorOriginMap.point.z << ","
                    << detections_[i]->header.stamp.toSec() << "\n";
         }catch(...){

         }
     }
    myfile.close();
    //pcl::io::savePCDFileASCII(filename,*getFrameTrack(frame));
}
