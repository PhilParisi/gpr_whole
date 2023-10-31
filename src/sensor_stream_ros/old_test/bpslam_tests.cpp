#include "../include/sensor_stream_ros/bpslam/bpslam.h"
#include "../include/sensor_stream_ros/bpslam/pf_base.hpp"
// Bring in gtest
#include <gtest/gtest.h>
#include <ros/ros.h>
// Declare a test
using namespace ss::bpslam;
TEST(PFBaseTests, constructor)
{
  PFBase<ParticleData> pf;
  EXPECT_NE(pf.rootParticlePtr(), nullptr);
}

TEST(PFBaseTests, addParticles)
{
  ss::bpslam::PFBase<ParticleData> pf;
//  pf.addNewLeaf(pf.leafQuequeFront());
//  pf.addNewLeaf(pf.leafQuequeFront());
//  pf.popLeaf();
  EXPECT_EQ(pf.rootParticlePtr()->getChildren().size(),2);
  EXPECT_EQ(pf.rootParticlePtr()->getChildren().size(),pf.getLeafQueue().size());
  EXPECT_EQ(pf.rootParticlePtr()->getParent(),nullptr); // root parent should be null
  //check that child nodes were created properly
  for (auto leaf : pf.rootParticlePtr()->getChildren()) {
    EXPECT_EQ(leaf->getParent(),pf.rootParticlePtr());  // parent should point to the root
    EXPECT_EQ(leaf->getChildren().size(),0);  // it should have no children
  }
  //try removing an element from the tree
  ss::bpslam::Particle<ParticleData>::Ptr particle_ptr =  *pf.rootParticlePtr()->getChildren().begin();
  particle_ptr->removeFromTree();
  EXPECT_EQ(pf.rootParticlePtr()->getChildren().size(),1);
  EXPECT_EQ(particle_ptr->getParent(),nullptr);
}

TEST(BPSlamTests, EKFWorker){
  ParticlePtr_t parent(new Particle_t);
  ParticlePtr_t child(new Particle_t);
  parent->addChild(child);

  nav_msgs::Odometry::Ptr odom_test(new nav_msgs::Odometry);
  for (size_t i=0;i<6;i++) {
    odom_test->twist.covariance[ss::idx::rowMajor(i,i,6)]=pow(1,2);
  }
  parent->getData()->nav.hypothesis.push_back(odom_test);

//  ss::bpslam::EKFWorker worker(child);
//  worker.run();
//  std::cout << "x: " << worker.output.velocity_error[ss::idx::x_linear] << std::endl;
//  std::cout << "y: " << worker.output.velocity_error[ss::idx::y_linear] << std::endl;
}




// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
  testing::InitGoogleTest(&argc, argv);
  ros::init(argc, argv, "PFBaseTests");
  return RUN_ALL_TESTS();
}
