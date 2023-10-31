#include<template_package/template_class.h>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "class_node_example");
  TemplateClass my_class;
  my_class.spin();
}
