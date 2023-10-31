#ifndef TEMPLATE_CLASS_H
#define TEMPLATE_CLASS_H
#include <ros/ros.h>
#include "std_msgs/String.h"

class TemplateClass
{
public:
  TemplateClass();
  /* \brief spin is degined to run the node continuously */
  void spin();
  /* \brief spin_once is intended to run the loop just once  */
  void spin_once();
  int returnOne();
  int returnTwo();

  int counter = 0;
  std_msgs::String message;

protected:
  ros::NodeHandlePtr ros_node_ptr_;
  ros::Publisher chatter_pub_;

};

#endif // TEMPLATE_CLASS_H
