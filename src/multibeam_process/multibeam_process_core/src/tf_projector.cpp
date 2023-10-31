#include "tf_projector.h"

namespace multibeam_process{

TfProjector::TfProjector()
{
  tf2_ros::Buffer(ros::Duration(1000000000,0));
  setUsingDedicatedThread(true);
}

void TfProjector::loadRobotModel(std::string filename){
  urdf::Model model;
  if (!model.initFile(filename)){
    ROS_ERROR("Failed to parse urdf file");
    return;
  }
  addRobotModel(model);
}

void TfProjector::addRobotModel(urdf::Model & model){
  robot_model_ = model;
  urdf::LinkConstSharedPtr link = model.getRoot();
  addStaticTf(link);
}


void TfProjector::addStaticTf(urdf::LinkConstSharedPtr link){
  std::vector<urdf::LinkSharedPtr> children = link->child_links;
  std::vector<urdf::JointSharedPtr> joints = link->child_joints;
  for(size_t i = 0 ; i <joints.size(); i++){
    if(joints[i]->type==urdf::Joint::FIXED){
      //std::cout << joints[i]->name << ": "<< joints[i]->parent_to_joint_origin_transform.position.x <<" "<<joints[i]->parent_to_joint_origin_transform.position.y <<" "<<joints[i]->parent_to_joint_origin_transform.position.z << std::endl;
      urdf::Pose pose = joints[i]->parent_to_joint_origin_transform;
      geometry_msgs::TransformStamped transform;
      //transform.header.stamp = startTime_;
      transform.header.frame_id = joints[i]->parent_link_name;
      transform.child_frame_id  = joints[i]->child_link_name;
      transform.transform.translation.x = pose.position.x;
      transform.transform.translation.y = pose.position.y;
      transform.transform.translation.z = pose.position.z;
      transform.transform.rotation.x = pose.rotation.x;
      transform.transform.rotation.y = pose.rotation.y;
      transform.transform.rotation.z = pose.rotation.z;
      transform.transform.rotation.w = pose.rotation.w;
      setTransform(transform,"tf_projector",true);
    }
    else{
      std::cout << "warning: TfProjector only supports fixed urdf joints" << std::endl;
    }
  }
  for(size_t i = 0 ; i <children.size(); i++){
    addStaticTf(children[i]);
  }
}

}
