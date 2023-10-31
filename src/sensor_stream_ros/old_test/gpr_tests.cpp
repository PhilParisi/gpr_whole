// Bring in my package's API, which is what I'm testing
#include "../include/sensor_stream_ros/gpr_bag_mapper.h"
// Bring in gtest
#include <gtest/gtest.h>
#include <ros/ros.h>
#include "../include/sensor_stream/gpr/sqexpsparse2d.h"
// Declare a test

TEST(BlockGPR, kernelV2Test){
  ss::kernels::SqExpSparse2d test_kernel;

  CudaMat<float> test_points(3,2);
  test_points.x(0)=0;test_points.y(0)=0;
  test_points.x(1)=1;test_points.y(1)=1;
  test_points.x(2)=10;test_points.y(2)=10;
  test_points.host2dev();

  CudaMat<float> hp(2,1);
  hp(0)=2;
  hp(1)=0.5;
  hp.host2dev();
  test_kernel.setHyperparams(hp);
  CudaMat<float> cov = test_kernel.genCovariance(test_points,test_points);
  cov.dev2host();
  cov.printHost();
  std::cout << cov.nnzCached() << std::endl;

  EXPECT_NEAR (0.5, cov(0,0), 1e-5);
  EXPECT_NEAR (0.00792875, cov(1,0), 1e-5);
  EXPECT_NEAR (0.00792875, cov(0,1), 1e-5);
  EXPECT_NEAR (0.5, cov(1,1), 1e-5);
  EXPECT_NEAR (0, cov(0,2), 1e-5);
  EXPECT_NEAR (0, cov(1,2), 1e-5);
  EXPECT_NEAR (0, cov(2,0), 1e-5);
  EXPECT_NEAR (0, cov(2,1), 1e-5);
  EXPECT_NEAR (.5, cov(2,2), 1e-5);
  EXPECT_EQ(cov.nnzCached(),5);

  CudaMat<float> sensor_var(3,1);
  sensor_var(0)=.1;
  sensor_var(1)=.2;
  sensor_var(2)=.3;
  sensor_var.host2dev();

  cov = test_kernel.genCovarianceWithSensorNoise(test_points,sensor_var);
  cov.dev2host();
  cov.printHost();

  EXPECT_NEAR (0.6, cov(0,0), 1e-5);
  EXPECT_NEAR (0.7, cov(1,1), 1e-5);
  EXPECT_NEAR (0.8, cov(2,2), 1e-5);

  std::cout<< "computing partial W.R.T length scale and comparing with mathematica"  << std::endl;
  CudaMat<float> d_l = test_kernel.partialDerivative(test_points,test_points,0);
  d_l.dev2host();
  d_l.printHost();
  for(size_t i = 0 ; i<3 ; i++)  // diagonal shoudl be zero
   EXPECT_NEAR (0, d_l(0,0), 1e-5);
  EXPECT_NEAR (0.0447032, d_l(1,0), 1e-5);
  EXPECT_NEAR (0.0447032, d_l(0,1), 1e-5);
  EXPECT_EQ(d_l.nnzCached(),2);

  std::cout<< "computing partial W.R.T process noise scale and comparing with mathematica"  << std::endl;
  CudaMat<float> d_sigma = test_kernel.partialDerivative(test_points,test_points,1);
  d_sigma.dev2host();
  d_sigma.printHost();
  for(size_t i = 0 ; i<3 ; i++)  // diagonal shoudl be zero
   EXPECT_NEAR (1, d_sigma(0,0), 1e-5);
  EXPECT_NEAR (0.0158575, d_sigma(1,0), 1e-5);
  EXPECT_NEAR (0.0158575, d_sigma(0,1), 1e-5);
  EXPECT_EQ(d_sigma.nnzCached(),5);
}

TEST(BlockGPR, partialsTest){
  BlockGpr gpr;
  gpr::GprParams params;
  params.reset();
  params.regression->kernel.reset(new ss::kernels::SqExpSparse2d);
  params.regression->kernel->hyperparam("length_scale")  = 6.1f;
  params.regression->kernel->hyperparam("process_noise") = 1.0f;
  params.regression->kernel->hyperparam2dev();
  params.regression->sensor_var = powf(0.2f,2);
  params.regression->nnzThresh=1e-16f;
  gpr.params = params;

  {
    CudaMat<float> x(2,2);
    x(0,0)=1    ;x(0,1)=0;
    x(1,0)=2    ;x(1,1)=0;
    x.host2dev();

    CudaMat<float> y(2,1);
    y(0,0)=1    ;
    y(1,0)=2    ;
    y.host2dev();
    gpr.addTrainingData(x,y);
  }
  {
    CudaMat<float> x(2,2);
    x(0,0)=1    ;x(0,1)=1;
    x(1,0)=2    ;x(1,1)=1;
    x.host2dev();

    CudaMat<float> y(2,1);
    y(0,0)=1.1    ;
    y(1,0)=2.2    ;
    y.host2dev();
    gpr.addTrainingData(x,y);
  }
  std::cout << "cholesky factor: " << std::endl;
  gpr.getCholeskyFactor().blocks2host();
  gpr.getCholeskyFactor().printHostValues();
  gpr.getCholeskyFactor().printHostValuesMathematica();

  std::cout << "gpr.derivativeLML(0) " << gpr.derivativeLML(0) << std::endl;
  std::cout << "gpr.derivativeLML(1) " << gpr.derivativeLML(1) << std::endl;

}

TEST(BlockGpr, logDetCholBlock)
{
  std::cout<< "computing the log of the determinant of the follwoing lower triangular cholesky factor "  << std::endl;
  std::cout<< "The results are compared with mathematica"  << std::endl;
  CudaMat<float> testMat(3,3);
  testMat(0,0)=1.3f;
  testMat(1,0)=18.2f;testMat(1,1)=2.8f;
  testMat(2,0)=14.3f;testMat(2,1)=0.01f;testMat(2,2)=0.0098f;

  testMat.printHost();
  testMat.host2dev();
  float result = BlockGpr::logDetCholBlock(testMat);
  std::cout<< "Log|k| = " << result << std::endl;
  EXPECT_NEAR (-6.66678, result, 1e-5);
}

TEST(BlockGpr, logDetChol)
{
  std::cout<< "Computing the log of the determinant of a lower triangular block cholesky factor"  << std::endl;
  std::cout<< "The results are compared with mathematica"  << std::endl;
  BlockGpr blockGpr;
  blockGpr.getCholeskyFactor().setBlockDim(3);
  {
    CudaMat<float> testMat(3,3);
    testMat(0,0)=1.3f;
    testMat(1,0)=18.2f;testMat(1,1)=2.8f;
    testMat(2,0)=14.3f;testMat(2,1)=0.01f;testMat(2,2)=0.0098f;
    testMat.host2dev();
    testMat.printHost();
    blockGpr.getCholeskyFactor().pushBackHost(testMat,0,0);
  }
  {
    CudaMat<float> testMat1(3,3);
    testMat1(0,2)=1;
    testMat1.host2dev();
    blockGpr.getCholeskyFactor().pushBackHost(testMat1,1,0);
  }
  {
    CudaMat<float> testMat2(3,3);
    testMat2(0,0)=2;
    testMat2(1,0)=0;testMat2(1,1)=3;
    testMat2(2,0)=0;testMat2(2,1)=0;testMat2(2,2)=4;
    testMat2.host2dev();
    blockGpr.getCholeskyFactor().pushBackHost(testMat2,1,1);
  }
  blockGpr.getCholeskyFactor().printHost();
  float result = blockGpr.logDetChol();
  std::cout<< "Log|k| = " << result << std::endl;
  EXPECT_NEAR (-0.310671, result, 1e-5);


  {
    CudaMat<float> y(3,1);
    blockGpr.getYTrain().setBlockParam(y.getDenseParam());
    for(size_t i = 0;i<3; i++){
      y(i,0)=i;
    }
    y.host2dev();
    y.printHost();
    blockGpr.getYTrain().pushBackHost(y,0,0);
  }
  {
    CudaMat<float> y(3,1);
    for(size_t i = 3;i<6; i++){
      y(i-3,0)=i;
    }
    y.host2dev();
    y.printHost();
    blockGpr.getYTrain().pushBackHost(y,1,0);
  }

  blockGpr.computeAlpha();
  //blockGpr.getAlpha().dev2host();
  std::cout<< "printing alpha: "<<   std::endl;
  blockGpr.getAlpha().printHost();
  blockGpr.getAlpha().getBlock(0,0).dev2host();
  blockGpr.getAlpha().getBlock(0,0).printHost();
  blockGpr.getAlpha().getBlock(1,0).dev2host();
  blockGpr.getAlpha().getBlock(1,0).printHost();
  float beta = blockGpr.lmlBetaTerm();
  std::cout<< "LML beta = " << beta << std::endl;
  EXPECT_NEAR (51576.0f, beta, 5e-2);
}




// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
  testing::InitGoogleTest(&argc, argv);
  ros::init(argc, argv, "BlockGpr");
  ros::NodeHandle nh;
  return RUN_ALL_TESTS();
}
