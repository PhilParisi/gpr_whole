
// Bring in header files to test
#include "../include/sensor_stream_ros/gpr_bag_mapper.h"
#include "../include/sensor_stream/cudamat.h"
#include "../include/sensor_stream/cudamat_operations.h"
// Bring in gtest
#include <gtest/gtest.h>
#include <ros/ros.h>

///////////////////////////////////////////////////////
// Tests for cudamat_operations.h
///////////////////////////////////////////////////////

// gtest resource http://google.github.io/googletest/reference/assertions.html
// location: sensor_stream_ros/sensor_stream/include/sensor_stream/cudamat_operations.h
// these functions interface with cublas_operations.h, namespace sensorstreamn --> ss::function_name()
// use case --> 1. define matrix, 2. matrix.host2dev(), 3. ss::function_name(matrix), 4. matrix.dev2host()

// none
TEST(CudamatOperationsTests, cudamat_operations_choleskyInvert)
{

}

// in progress
TEST(CudamatOperationsTests, cudamat_operations_choleskyDecompose)
{
  //Testing ss::choleskyDecompose
  // mathematical answer provided by MATLAB cudamat_operations_testcases.m
  CudaMat<float> A_mat(3,3);
  A_mat(0,0) = 6; A_mat(0,1) = 15; A_mat(0,2) = 55;
  A_mat(1,0) = 15; A_mat(1,1) = 55; A_mat(1,2) = 225;
  A_mat(2,0) = 55; A_mat(2,1) = 225; A_mat(2,2) = 979;

  //A_mat.printHost();

  A_mat.host2dev();
  float chol_answer;
  chol_answer = ss::choleskyDecompose(A_mat, CUBLAS_FILL_MODE_LOWER);
  A_mat.dev2host();

  //A_mat.printHost();

  // need to work this a bit more
  // choleskyDecompose() doesnt give us a pure lower triangular matrix (why?)
  // wnat to check the different arguments to choleskyDecompose (fill mode lower / fill mode upper)
  // check the chol_answer when successful and when unsuccessful

}

// none
TEST(CudamatOperationsTests, cudamat_operations_triangleSolve)
{

}

// none
TEST(CudamatOperationsTests, cudamat_operations_choleskySolve)
{

}

// semi-done
TEST(CudamatOperationsTests, cudamat_operations_multDiagonal)
{
  //Testing ss::multDiagonal
  // multDiagonal computes the diagonal element of the result of a*b as a nx1

  //Agreeing Dimensions
  CudaMat<float> A_mat(2,3),B_mat(3,2);
  A_mat(0,0) = 54.9f; A_mat(0,1) = 48.2f; A_mat(0,2) = 99.1f;
  A_mat(1,0) = -17.9f; A_mat(1,1) = -5.98f; A_mat(1,2) = -1.9f;
  B_mat(0,0) = 9.2f; B_mat(0,1) = -2.2f;
  B_mat(1,0) = -4.0f; B_mat(1,1) = -2.2f;
  B_mat(2,0) = 25.0f; B_mat(2,1) = 3.7f;
  A_mat.printMathematica();
  CudaMat<float> A_check = A_mat.deepCopy();
  CudaMat<float> B_check = B_mat.deepCopy();

  A_mat.host2dev(); B_mat.host2dev();
  CudaMat<float> C_mat = ss::multDiagonal(A_mat,B_mat);
  A_mat.dev2host(); B_mat.dev2host(); C_mat.dev2host();

  CudaMat<float> matlab_ans(2,1);
  matlab_ans(0,0) = 2789.780f; matlab_ans(1,0) = 45.506f;

  //check values
  EXPECT_FLOAT_EQ(C_mat(0,0),matlab_ans(0,0));
  EXPECT_FLOAT_EQ(C_mat(1,0),matlab_ans(1,0));
  //check size
  EXPECT_EQ(C_mat.rows(),std::min(A_check.rows(),B_check.rows()));

  // TODO test failure mode, matrices of disagreeing dimensions
}

// done
TEST(CudamatOperationsTests, cudamat_operations_elementWiseDivide)
{
  //Testing elementWiseDivide()
  // A(i,j) divided by B(i,j), overwrites A
  CudaMat<float> A_mat(2,3),B_mat(2,3);
  A_mat(0,0) = 54.9; A_mat(0,1) = 48.2; A_mat(0,2) = -60.3;
  A_mat(1,0) = -17.9; A_mat(1,1) = -5.98; A_mat(1,2) = 9.48;
  B_mat(0,0) = 9.2; B_mat(0,1) = -2.2; B_mat(0,2) = 77.8;
  B_mat(1,0) = -4.0; B_mat(1,1) = -2.2; B_mat(1,2) = 2.35;

  CudaMat<float> A_check = A_mat.deepCopy();
  CudaMat<float> B_check = B_mat.deepCopy();

  A_mat.host2dev(); B_mat.host2dev();
  ss::elementWiseDivide(A_mat, B_mat);
  A_mat.dev2host(); B_mat.dev2host();

  //check A changed PHIL FOR LOOP to CHECK EACH
  // EXPECT_FLOAT_EQ
  EXPECT_FLOAT_EQ(A_mat(0,0),A_check(0,0)/B_check(0,0));
  EXPECT_EQ(A_mat(0,1),A_check(0,1)/B_check(0,1));
  EXPECT_EQ(A_mat(0,2),A_check(0,2)/B_check(0,2));
  EXPECT_EQ(A_mat(1,0),A_check(1,0)/B_check(1,0));
  EXPECT_EQ(A_mat(1,1),A_check(1,1)/B_check(1,1));
  EXPECT_EQ(A_mat(1,2),A_check(1,2)/B_check(1,2));
  //check B is same
  EXPECT_EQ(B_mat(0,0),B_check(0,0));
  EXPECT_EQ(B_mat(0,1),B_check(0,1));
  EXPECT_EQ(B_mat(0,2),B_check(0,2));
  EXPECT_EQ(B_mat(1,0),B_check(1,0));
  EXPECT_EQ(B_mat(1,1),B_check(1,1));
  EXPECT_EQ(B_mat(1,2),B_check(1,2));
}

// done
TEST(CudamatOperationsTests, cudamat_operations_getDiagonal)
{
  //Testing ss::getDiagonal
  // puts diagonal elements into new nx1 cudamat
  CudaMat<float> A_mat(2,2);
//  A_mat(0,0) = 5.0f;  A_mat(0,1) = 29.6f;
//  A_mat(1,0) = -1025.3f; A_mat(1,1) = -4.0f;
  for(size_t i = 0; i< A_mat.size(); i++){
    A_mat(i) = float(i);
  }


  A_mat.host2dev();
  CudaMat<float> A_diag = ss::getDiagonal(A_mat);
  A_mat.dev2host();
  A_diag.dev2host(); // A_diag defined on the GPU, need to pass dev2host()

  //check values FOR LOOP
  EXPECT_EQ(A_diag(0,0),A_mat(0,0));
  EXPECT_EQ(A_diag(1,0),A_mat(1,1));
  //check num rows and columns
  EXPECT_EQ(A_diag.rows(), A_mat.rows());
  EXPECT_EQ(A_diag.cols(), 1);
}

// done
TEST(CudamatOperationsTests, cudamat_operations_setConstant)
{
  //Testing ss::setConstant()
  // setConstant should makes all the matrix values the argument to the function
  CudaMat<float> A_mat(2,2);
  A_mat(0,0) = 5.0;  A_mat(0,1) = 29.6;
  A_mat(1,0) = -1025.3; A_mat(1,1) = -4.0;
  float alpha = 8.0;

  A_mat.host2dev();
  ss::setConstant(A_mat,alpha); // operation on GPU (device)
  A_mat.dev2host();

  EXPECT_EQ(A_mat(0,0),alpha);
  EXPECT_EQ(A_mat(0,1),alpha);
  EXPECT_EQ(A_mat(1,0),alpha);
  EXPECT_EQ(A_mat(1,1),alpha);
}

// done, krasno trace question
TEST(CudamatOperationsTests, cudamat_operations_trace)
{
  //Testing trace()
  // trace in linear algebra = sum of elements along main diagonal (top left to bottom right)

  // Square Matrix Test
  CudaMat<float> A_mat(2,2); // 2x2 CudaMat
  A_mat(0,0) = 5.0;  A_mat(0,1) = 12.0;
  A_mat(1,0) = 12.0; A_mat(1,1) = -40.0;

  A_mat.host2dev();
  float A_mat_trace = ss::trace(A_mat);   // takes place on GPU
  A_mat.dev2host();

  float manual_trace;
  manual_trace = A_mat(0,0) + A_mat(1,1);
  EXPECT_EQ(A_mat_trace,manual_trace);

  // Non-Square Matrix Test [trace, in principal, is only for square matrices... trace WORKS for non-square matrices]
  CudaMat<float> B_mat(2,3); // 2x3 CudaMat
  B_mat(0,0) = 9.2; B_mat(0,1) = -2.2; B_mat(0,2) = 77.8;
  B_mat(1,0) = -4.0; B_mat(1,1) = -2.2; B_mat(1,2) = 2.35;

  B_mat.host2dev();
  float B_mat_trace = ss::trace(B_mat);
  B_mat.dev2host();
  std::cout << "The trace is: " << B_mat_trace << std::endl;
}



































///
/// kris's old tests below
///

/*
// Declare a test
TEST(CudamatOperationsTests, addConstant)
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
TEST(CudamatOperationsTests, choleskySolveTranspose)
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
