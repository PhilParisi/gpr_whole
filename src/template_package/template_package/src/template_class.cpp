#include <template_package/template_class.h>

TemplateClass::TemplateClass()
{
  /* I decided to use a pointer to my ros node no I need to allocate memory
   * also,  I told it to use the the node's namespace (specified by ~ just like linux directories)
   * basically all that means is all broadcast topics will be sent as "node_name/topic_name" unless
   * otherwise specified. */
  ros_node_ptr_.reset(new ros::NodeHandle("~"));
  chatter_pub_ = ros_node_ptr_->advertise<std_msgs::String>("chatter", 1000);
}

void TemplateClass::spin_once(){
  std::stringstream ss;
  ss << "hello world " << counter;
  message.data = ss.str();

  ROS_INFO("%s", message.data.c_str());
  chatter_pub_.publish(message);
  ++counter;
  // after we spin once we need to tell ros to do the same
  ros::spinOnce();
}

void TemplateClass::spin(){
  ros::Rate loop_rate(1); // in hz
  while (ros::ok())
  {
    spin_once();
    loop_rate.sleep();
  }
}

int TemplateClass::returnOne(){
  return 1;
}

int TemplateClass::returnTwo(){
  return 1;
}
