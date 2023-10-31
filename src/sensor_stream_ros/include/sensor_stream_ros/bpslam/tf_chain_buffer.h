#ifndef TF_CHAIN_BUFFER_H
#define TF_CHAIN_BUFFER_H
#include <tf2_ros/buffer.h>

namespace ss{ namespace bpslam {

class TfChainBuffer: public tf2_ros::Buffer, std::enable_shared_from_this<TfChainBuffer>
{
public:
  typedef std::shared_ptr<TfChainBuffer> Ptr;
  typedef std::shared_ptr<const TfChainBuffer> ConstPtr;
  TfChainBuffer();
  TfChainBuffer::Ptr getThis(){return shared_from_this();}
  TfChainBuffer::Ptr getParent() const {return parent_;}
  void setParent(TfChainBuffer::Ptr parent);//{parent_=parent;}
  bool isRoot() const {return parent_==nullptr;}

  bool inTimeRange(const ros::Time& time) const;
  bool betweenParentBuffer(const ros::Time& time) const;

  bool setTransform(const geometry_msgs::TransformStamped& transform,
                    const std::string & authority, bool is_static = false);

  geometry_msgs::TransformStamped
  lookupTransform(const std::string& target_frame, const std::string& source_frame,
                  const ros::Time& time, const ros::Duration timeout) const;


protected:
  std::shared_ptr<TfChainBuffer> parent_;
  TfChainBuffer::ConstPtr const_this_;
  geometry_msgs::TransformStamped earliest_transform_;
  geometry_msgs::TransformStamped latest_trasnform_;
};

}}
#endif // TF_CHAIN_BUFFER_H
