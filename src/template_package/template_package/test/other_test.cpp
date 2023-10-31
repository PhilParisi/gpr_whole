// Bring in my package's API, which is what I'm testing
#include "template_package/template_class.h"
// Bring in gtest
#include <gtest/gtest.h>
#include <ros/ros.h>

// Declare a test
TEST(TestSuite, testCase1)
{
  TemplateClass test_class;
  EXPECT_EQ(1, test_class.returnOne());
}

TEST(TestSuite, testCase2)
{
  TemplateClass test_class;
  EXPECT_EQ(2, test_class.returnTwo());
}


// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
  testing::InitGoogleTest(&argc, argv);
  ros::init(argc, argv, "tester");
  ros::NodeHandle nh;
  return RUN_ALL_TESTS();
}
