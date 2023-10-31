#include <ros/ros.h>
#include "std_msgs/String.h"
#include <sstream>
#include <template_package/template_class.h>  /* this should bring in the TemplateClass */


/** this file is meant to be object oriented version of hello_node.cpp */
/** goal is to create an object that has an attribute that gets updated, and that attribute is published */


int main(int argc, char **argv)
{

  ros::init(argc, argv, "talker");
  ros::NodeHandle n;
  ros::Publisher chatter_pub = n.advertise<std_msgs::String>("chatter", 1000);
  ros::Rate loop_rate(10);

  TemplateClass TemplatePublisher;

  while (ros::ok())
  {
    /** build the string, put it into ojbect, publish object message
     */
    std::stringstream ss;
    ss << "hello world " << TemplatePublisher.counter;
    TemplatePublisher.message.data = ss.str();

    ROS_INFO("%s", TemplatePublisher.message.data.c_str());
    chatter_pub.publish(TemplatePublisher.message);

    ros::spinOnce();

    loop_rate.sleep();
    ++TemplatePublisher.counter;
  }


  return 0;
}
