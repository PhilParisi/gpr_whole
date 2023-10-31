
// Bring in header files to test
#include "../include/sensor_stream_ros/gpr_bag_mapper.h"
#include "../include/sensor_stream/cudamat.h"
// Bring in libraries
#include <gtest/gtest.h>
#include <ros/ros.h>
#include <cstdlib>
#include <iostream>
#include <ctime>

///////////////////////////////////////////////////////
// Tests for cudamat.h
///////////////////////////////////////////////////////

// gtest resource http://google.github.io/googletest/reference/assertions.html
// location: sensor_stream_ros/sensor_stream/include/sensor_stream/cudamat.h
// cudamat.h defines the basis for the CudaMat Class. Hence, we're testing basic instantiation features.

TEST(CudamatTests, enum_memType)
{
  // Testing enum memType
  memType A_memType = host;
  memType B_memType = dev;
  EXPECT_EQ(A_memType, 0);
  EXPECT_EQ(B_memType, 1);
}

TEST(CudamatTests, denseParam_t)
{
  // Testing denseParam_t basic constructor
  denseParam_t basic_paramA;
  size_t alpha = 7; size_t beta = 9;
  basic_paramA.rows = alpha;
  basic_paramA.cols = beta;
  EXPECT_EQ(basic_paramA.getRows(CUBLAS_OP_N),alpha);
  EXPECT_EQ(basic_paramA.getCols(CUBLAS_OP_N),beta);

  // Testing denseParam_t constructor w/ row,col
  denseParam_t param(1,2);
  EXPECT_EQ(1,param.rows);
  EXPECT_EQ(2,param.cols);
  EXPECT_EQ(param.rows,param.getRows(CUBLAS_OP_N));
  EXPECT_EQ(param.cols,param.getCols(CUBLAS_OP_N));

  // Testing overloaded comparison operator ==
  denseParam_t other_param(2,1);
  EXPECT_FALSE(other_param==param);
  other_param=param; //set objects equal
  EXPECT_TRUE(other_param==param);

  // Testing getRows method and switch case
  EXPECT_EQ(1, param.getRows(CUBLAS_OP_N));
  EXPECT_EQ(2, param.getRows(CUBLAS_OP_T));
  EXPECT_EQ(2, param.getRows(CUBLAS_OP_C));

  // Testing getCols method and switch case
  EXPECT_EQ(2, param.getCols(CUBLAS_OP_N));
  EXPECT_EQ(1, param.getCols(CUBLAS_OP_T));
  EXPECT_EQ(1, param.getCols(CUBLAS_OP_C));

  // Testing rows and cols member variables [Line 97]
  EXPECT_EQ(1, param.rows);
  EXPECT_EQ(2, param.cols);
}

TEST(CudamatTests, metadata_t)
{
  // Testing metadata_t
  metadata_t meta;
  EXPECT_EQ(-1, meta.nnzCached);
  EXPECT_EQ(false, meta.is_zero);
  meta.nnzCached = 10;
  EXPECT_EQ(10, meta.nnzCached);
  meta.is_zero = true;
  EXPECT_EQ(true, meta.is_zero);
}

TEST(CudamatTests, CudaMat_constructors)
{
  // Testing Cudamat Class Constructors
  srand(42);
  size_t alpha = rand() % 100 + 1;
  size_t beta = rand() % 100 + 1;

  // with rows, cols, memType default
  CudaMat<float> matB(alpha,beta);
  EXPECT_EQ(alpha, matB.rows());
  EXPECT_EQ(beta, matB.cols());
  EXPECT_EQ(alpha*beta, matB.size());
  EXPECT_EQ(alpha, matB.ld());
  EXPECT_EQ(matB.rows(), matB.ld());
  // want to check the memType = host (0), but how to access that? shared_ptr?

  // with denseParam_t
  denseParam_t params(alpha,beta);
  CudaMat<float> matC(params);
  EXPECT_EQ(alpha, matC.rows());
  EXPECT_EQ(beta, matC.cols());
  EXPECT_EQ(alpha*beta, matC.size());
  EXPECT_EQ(alpha, matC.ld());
  EXPECT_EQ(matC.rows(), matC.ld());

  // with cudamat
  CudaMat<float> matA(matC);
  EXPECT_EQ(matA.rows(), matC.rows());
  EXPECT_EQ(matA.cols(), matC.cols());
  EXPECT_EQ(matA.size(), matC.size());
  EXPECT_EQ(matA.ld(), matC.ld());
  EXPECT_EQ(matA.rows(), matA.ld());
}

TEST(CudamatTests, CudaMat_memory_management_voids)
{
  // Testing CudaMat Memory Management Functions
  CudaMat<float> matB;
  srand(42);
  size_t alpha = rand() % 100 + 1;
  size_t beta = rand() % 100 + 1;
  size_t gamma = rand() % 100 + 1;

  // Testing reset()
  matB.reset(alpha,beta);
  EXPECT_EQ(alpha, matB.rows());
  EXPECT_EQ(beta, matB.cols());
  EXPECT_EQ(alpha*beta, matB.size());
  EXPECT_EQ(alpha, matB.ld());
  EXPECT_EQ(matB.rows(), matB.ld());

  // Testing initDev()
  matB.initDev();
  EXPECT_EQ(alpha, matB.rows());
  EXPECT_EQ(beta, matB.cols());
  EXPECT_EQ(alpha*beta, matB.size());
  EXPECT_EQ(alpha, matB.ld());
  EXPECT_EQ(matB.rows(), matB.ld());
  // is there a parameter (memType) we can check for it's location? dev/host?
  // worth testing a matB.printDev()? how to test the output of that, simply that is doesn't throw an exception?

  // Testing initDevIdentity()
  CudaMat<float> matA(alpha,beta);
  matA.initDevIdentity(gamma);
  matA.dev2host();
  EXPECT_EQ(matA.rows(), matA.cols());
  EXPECT_EQ(matA.rows(), gamma);
  EXPECT_EQ(gamma*gamma, matA.size());
  EXPECT_EQ(matA.ld(), gamma);
  for (size_t i = 0; i < matA.rows(); i++)
  {
    for (size_t j = 0; j < matA.cols(); j++)
    {
      if (i == j)
        EXPECT_EQ(1,matA(i,j));
      else
        EXPECT_EQ(0,matA(i,j));
    }
  }

  // Testing initHost()
  CudaMat<float> matC(alpha, gamma);
  matC.initHost();
  EXPECT_EQ(alpha, matC.rows());
  EXPECT_EQ(gamma, matC.cols());
  EXPECT_EQ(alpha*gamma, matC.size());
  EXPECT_EQ(alpha, matC.ld());
  EXPECT_EQ(matC.rows(), matC.ld());

  // Testing host2dev() and dev2host()
  CudaMat<float> matD(1,3);
  matD(0,0) = alpha; matD(0,1) = beta; matD(0,2) = gamma;
  CudaMat<float> D_check = matD.deepCopy();
  matD.host2dev();
  mult(matD,gamma); // elementwise (IS THIS A DEVICE or HOST OPERATION?)
  matD.dev2host();

  for (size_t k = 0; k < matD.size(); k++)
    EXPECT_EQ(matD(k),D_check(k)*gamma);
}

TEST(CudamatTests, CudaMat_memory_management_deepcopy)
{
  //Testing deepCopy()
  CudaMat<float> shallow, deep;
  CudaMat<float> matX(2,2);
  shallow = matX;
  float alpha = 88.29f;
  matX(0,0) = alpha;
  EXPECT_FLOAT_EQ(matX(0,0), shallow(0,0));
  deep = matX.deepCopy();
  float beta = 111.01f;
  matX(0,0) = beta;
  EXPECT_FLOAT_EQ(deep(0,0), alpha);
  EXPECT_FLOAT_EQ(matX(0,0), beta);
  EXPECT_FLOAT_EQ(shallow(0,0), beta);
}

// LEFT OFF HERE on 8Feb2022
TEST(CudamatTests, CudaMat_dimensionality)
{
  //Testing Dimensionality
  // rows,cols,ld,size, tested previously
  // reserveDev()
  // addRowsDev()
  // addColsDev()
  // insertDev()
}

TEST(CudamatTests, CudaMat_accessors)
{
  //Testing Accessors

}

TEST(CudamatTests, CudaMat_operatoroverload)
{
  //Testing Operators [Line 272]
  // WHY ARE THESE NOT WORKING???
  // +=
  CudaMat<float> cmA(1,4), cmB(4,1), cmC(2,4);
  /*CudaMat<float> cmJ(1,4);
  cmJ = cmA += cmA; // 1x4 + 1x4 = 1x4
  EXPECT_EQ(1,cmJ.rows());
  EXPECT_EQ(4,cmJ.cols());
  // -=
  CudaMat<float> cmK(1,4);
  cmK = cmA -= cmA; // 1x4 - 1x4 = 1x4
  EXPECT_EQ(cmK.rows(),cmJ.rows());
  EXPECT_EQ(cmK.cols(),cmK.cols());
  //
  //CudaMat<float> cmP(2,1);
  //cmP = cmC * cmB; // 2x4 * 4x1 = 2x1
  //EXPECT_EQ(2,cmP.rows());
  //EXPECT_EQ(1,cmP.cols());
  */
}

































///
/// kris's old tests below
///

/*
// Declare a test
TEST(CudamatTests, addConstant)
{
  CudaMat<float> testmat(5,5,dev);
  add(testmat,5);
  testmat.dev2host();
  testmat.printHost();
  for (size_t i =0 ;i<testmat.rows();i++) {
    for (size_t j =0 ;j<testmat.cols();j++) {
      EXPECT_NEAR(testmat(i,j),5.0f,1e-5);
    }
  }
}
TEST(CudamatTests, choleskySolveTranspose)
{
  std::cout << "in this test we check the operations involved with cholesky solve on a cudablockmat" << std::endl;
  CudaBlockMat<float> L;
  denseParam_t block_params(2,2);
  L.setBlockParam(block_params);

  CudaMat<float> L_0_0(block_params,host);
  L_0_0(0,0)=1;  L_0_0(0,1)=0;
  L_0_0(1,0)=5;  L_0_0(1,1)=2;
  std::cout << "submatrix L(0,0): " << std::endl;
  L_0_0.printHost();
  //L_0_0.host2dev();
  L.pushBackHost(L_0_0,0,0);

  CudaMat<float> L_1_0(block_params,host);
  L_1_0(0,0)=8;  L_1_0(0,1)=6;
  L_1_0(1,0)=10;  L_1_0(1,1)=9;
  std::cout << "submatrix L(1,0): " << std::endl;
  L_1_0.printHost();
  //L_1_0.host2dev();
  L.pushBackHost(L_1_0,1,0);

  CudaMat<float> L_1_1(block_params,host);
  L_1_1(0,0)=3;  L_1_1(0,1)=0;
  L_1_1(1,0)=7;  L_1_1(1,1)=4;
  std::cout << "submatrix L(1,1): " << std::endl;
  L_1_1.printHost();
  //L_1_1.host2dev();
  L.pushBackHost(L_1_1,1,1);

  L.printHostValues();

  L.blocks2dev();


  CudaBlockMat<float> y;
  y.setBlockParam(block_params);

  CudaMat<float> y_0_0(block_params,host);
  y_0_0(0,0)=1;  y_0_0(0,1)=2;
  y_0_0(1,0)=3;  y_0_0(1,1)=4;
  y.pushBackHost(y_0_0,0,0);

  CudaMat<float> y_1_0(block_params,host);
  y_1_0(0,0)=5;  y_1_0(0,1)=6;
  y_1_0(1,0)=7;  y_1_0(1,1)=8;
  y.pushBackHost(y_1_0,1,0);

  y.blocks2dev();
  auto x=y.deepCopy();

  std::cout<< "Cholesky Solve: "<<   std::endl;
  ss::choleskySolve(L,x);
  x.blocks2host();
  x.printHostValues();


  CudaBlockMat<float> y2;
  y2.setBlockParam(block_params);

  CudaMat<float> y2_0_0(block_params,host);
  y2_0_0(0,0)=1;  y2_0_0(0,1)=3;
  y2_0_0(1,0)=2;  y2_0_0(1,1)=4;
  y2.pushBackHost(y2_0_0,0,0);

  CudaMat<float> y2_0_1(block_params,host);
  y2_0_1(0,0)=5;  y2_0_1(0,1)=7;
  y2_0_1(1,0)=6;  y2_0_1(1,1)=8;
  y2.pushBackHost(y2_0_1,0,1);

  y2.printHostValues();
  std::cout << std::endl;

  y2.blocks2dev();
  auto y3 = y2.deepCopy();

  ss::ltSolve(L,y2,CUSPARSE_OPERATION_TRANSPOSE,CUSPARSE_OPERATION_TRANSPOSE);

  y2.blocks2host();
  std::cout<< "Upper Triangular Solve: "<<   std::endl;
  y2.printHostValues();

  ss::ltSolve(L,y3,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_TRANSPOSE);

  std::cout<< "Lower Triangular Solve: "<<   std::endl;
  y3.blocks2host();
  y3.printHostValues();


  CudaMat<float> y_squared = ss::singleRowMult(y,y,0,CUSPARSE_OPERATION_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE);

  std::cout<< "Y_squared: "<<   std::endl;
  y_squared.dev2host();
  y_squared.printHost();


}
*/

// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
  testing::InitGoogleTest(&argc, argv);
  //ros::init(argc, argv, "tester");
  //ros::NodeHandle nh;
  return RUN_ALL_TESTS();
}
