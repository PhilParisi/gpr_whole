#ifndef TF_PROJECTOR_H
#define TF_PROJECTOR_H

#include <urdf/model.h>
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/message_filter.h"
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_msgs/TFMessage.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <geometry_msgs/TransformStamped.h>
#include <memory>

namespace multibeam_process{
class TfProjector: public tf2_ros::Buffer
{
public:
  TfProjector();
  /*!
   * \brief set the frame you want to treat as fixed
   * \param mapFrame the fixed frame
   */
  void setMapFrame(std::string map_frame){map_frame_=map_frame;}
  void loadRobotModel(std::string filename);
  /*!
   * \brief Add a urdf model to the TF tree as a static TF
   * \param model
   */
  void addRobotModel(urdf::Model &model);
  /*!
   * \brief add a static tf frame from a from a URDF Link
   * \param link
   */
  void addStaticTf(urdf::LinkConstSharedPtr link);

  /*!
   * \brief returns a copy of the robot model
   * \return a copy of the robot model
   */
  urdf::Model getRobotModel(){return robot_model_;}

  typedef std::shared_ptr<TfProjector> Ptr;


private:
  urdf::Model robot_model_;
  std::string map_frame_;
};
}
#endif // TF_PROJECTOR_H
