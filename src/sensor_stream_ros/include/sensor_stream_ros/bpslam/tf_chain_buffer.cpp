#ifndef TF_CHAIN_BUFFER_HPP
#define TF_CHAIN_BUFFER_HPP
#include "tf_chain_buffer.h"
namespace ss{ namespace bpslam {
TfChainBuffer::TfChainBuffer() : tf2_ros::Buffer(ros::Duration(1000000000,0),false)
{
  setUsingDedicatedThread(true);
}

bool TfChainBuffer::inTimeRange(const ros::Time &time) const{
  return time>=earliest_transform_.header.stamp
      && time<=latest_trasnform_.header.stamp;
}

bool TfChainBuffer::betweenParentBuffer(const ros::Time &time) const{
  return time<earliest_transform_.header.stamp
      && time>getParent()->latest_trasnform_.header.stamp;
}

bool TfChainBuffer::setTransform(const geometry_msgs::TransformStamped &transform, const std::string &authority, bool is_static){
  if(earliest_transform_.header.stamp.sec == 0){
    earliest_transform_=transform;
  }
  if(transform.header.stamp<earliest_transform_.header.stamp){
    earliest_transform_=transform;
  }
  if(transform.header.stamp>latest_trasnform_.header.stamp){
    latest_trasnform_=transform;
  }
  return tf2_ros::Buffer::setTransform(transform,authority,is_static);
}

geometry_msgs::TransformStamped
TfChainBuffer::lookupTransform(const std::string& target_frame, const std::string& source_frame,
                               const ros::Time& time, const ros::Duration timeout) const{
  geometry_msgs::TransformStamped out;

  const TfChainBuffer * current_buffer = this;
  bool failed=true;
  while(failed){
    if(current_buffer->inTimeRange(time)){
      out = current_buffer->tf2_ros::Buffer::lookupTransform(target_frame,source_frame,time,timeout);
      failed=false;
    }else if(current_buffer->betweenParentBuffer(time)){
      out = earliest_transform_;
      failed=false;
    }else{
      if(current_buffer->isRoot()){
        throw(tf2::ExtrapolationException("searched all ancestor tf trees.  no valid tf found"));
      }
      current_buffer=current_buffer->getParent().get();
    }
  }

  return out;
}

void TfChainBuffer::setParent(TfChainBuffer::Ptr parent){
  parent_=parent;

}
}}
#endif
