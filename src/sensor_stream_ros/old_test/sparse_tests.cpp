// Bring in my package's API, which is what I'm testing
#include "../include/sensor_stream_ros/gpr_bag_mapper.h"
// Bring in gtest
#include <gtest/gtest.h>
#include <ros/ros.h>
#include <sensor_stream/include/sensor_stream/cudablockmatcsc.h>
// Declare a test

TEST(SparseTests, choleskySolve){
  CudaBlockMat<float> L;
  L.setBlockDim(2);
  {
    CudaMat<float> testMat(2,2);
    testMat(0,0)=1    ;testMat(0,1)=0;
    testMat(1,0)=2    ;testMat(1,1)=3;
    testMat.host2dev();
    L.pushBackHost(testMat,0,0);
  }
  {
    CudaMat<float> testMat(2,2);
    testMat(0,0)=4    ;testMat(0,1)=5;
    testMat(1,0)=6    ;testMat(1,1)=7;
    testMat.host2dev();
    L.pushBackHost(testMat,1,0);
  }
  {
    CudaMat<float> testMat(2,2);
    testMat(0,0)=8    ;testMat(0,1)=0;
    testMat(1,0)=9    ;testMat(1,1)=10;
    testMat.host2dev();
    L.pushBackHost(testMat,1,1);
  }
  {
    CudaMat<float> testMat(2,2);
    testMat(0,0)=12    ;testMat(0,1)=13;
    testMat(1,0)=14    ;testMat(1,1)=15;
    testMat.host2dev();
    L.pushBackHost(testMat,2,1);
  }
  {
    CudaMat<float> testMat(2,2);
    testMat(0,0)=16    ;testMat(0,1)=0;
    testMat(1,0)=17    ;testMat(1,1)=18;
    testMat.host2dev();
    L.pushBackHost(testMat,2,2);
  }
  std::cout << std::endl << "L" << std::endl;
  L.printHostValues();

  CudaBlockMat<float> B;
  B.setBlockDim(2);
  {
    CudaMat<float> testMat(2,2);
    testMat(0,0)=1    ;testMat(0,1)=2;
    testMat(1,0)=3    ;testMat(1,1)=4;
    testMat.host2dev();
    B.pushBackHost(testMat,0,0);
  }
  {
    CudaMat<float> testMat(2,2);
    testMat(0,0)=5    ;testMat(0,1)=6;
    testMat(1,0)=7    ;testMat(1,1)=8;
    testMat.host2dev();
    B.pushBackHost(testMat,2,0);
  }
  {
    CudaMat<float> testMat(2,2);
    testMat(0,0)=9    ;testMat(0,1)=10;
    testMat(1,0)=11    ;testMat(1,1)=12;
    testMat.host2dev();
    B.pushBackHost(testMat,2,1);
  }

  std::cout << std::endl << "B" << std::endl;
  B.printHostValues();
  auto B_copy = B.deepCopy();

  ss::ltSolveMat(L,B);
  std::cout << std::endl << "ss::ltSolveMat(L,B)" << std::endl;
  B.blocks2host();
  B.printHostValues();

  {
  std::cout << std::endl << "Comparing ss::ltSolveMat(L,B) to the mathematica solution..." << std::endl;
  std::vector<std::vector<float>> mathematica_soln = {{1., 2., 0., 0.}, {0.333333, 0., 0., 0.}, {-0.708333, -1., 0., 0.}, {-0.195833, -0.3, 0., 0.}, {1.00286, 1.36875, 0.5625, 0.625}, {0.155859, 0.179514, 0.0798611, 0.0763889}};
  for(size_t i = 0; i<B.valuesPerRow(); i++){
    for(size_t j = 0; j<B.valuesPerCol(); j++){
      EXPECT_NEAR(B.getVal(i,j),mathematica_soln[i][j],1e-5);
    }
  }
  }




  std::cout << std::endl << "B_copy" << std::endl;
  B_copy.blocks2host();
  B_copy.printHostValues();

  ss::choleskySolveMat(L,B_copy);
  std::cout << std::endl << "choleskySolveMat(L,B_copy)" << std::endl;
  B_copy.blocks2host();
  B_copy.printHostValues();

  {
  std::cout << std::endl << "Comparing choleskySolveMat(L,B_copy) to the mathematica solution..." << std::endl;
  std::vector<std::vector<float>> mathematica_soln =    {{0.959942,   2.25298,    0.0625859,  0.0694209},
                                                         {0.464426,   0.490062,   0.110235,   0.122267},
                                                         {-0.0690571, -0.094684, -0.00141888, -0.00156099},
                                                         {-0.102094,  -0.142395, -0.04623,    -0.0512852},
                                                         {0.053479,   0.0749506,  0.0304422,  0.0345534},
                                                         {0.00865885, 0.00997299, 0.00443673, 0.00424383}};
  for(size_t i = 0; i<B_copy.valuesPerRow(); i++){
    for(size_t j = 0; j<B_copy.valuesPerCol(); j++){
      EXPECT_NEAR(B_copy.getVal(i,j),mathematica_soln[i][j],1e-5);
    }
  }
  }




  L=L.transposeHost();
  std::cout << std::endl << "L.transposeHost()" << std::endl;
  L.blocks2host();
  L.printHostValues();
  ss::utSolveMat(L,B,CUBLAS_OP_T);  // we transposed L so we need to treat it's blocks as transpose
  std::cout << std::endl << "ss::utSolveMat(L,B,CUBLAS_OP_T)" << std::endl;
  B.blocks2host();
  B.printHostValues();
  {
  std::cout << std::endl << "Comparing ss::utSolveMat(L,B,CUBLAS_OP_T) to the mathematica solution..." << std::endl;
  std::vector<std::vector<float>> mathematica_soln =   {{0.959942, 2.25298, 0.0625859, 0.0694209}, {0.464426, 0.490062,0.110235, 0.122267}, {-0.0690571, -0.094684, -0.00141888, -0.00156099}, {-0.102094, -0.142395, -0.04623, -0.0512852}, {0.053479, 0.0749506, 0.0304422, 0.0345534}, {0.00865885, 0.00997299, 0.00443673, 0.00424383}};
  for(size_t i = 0; i<B.valuesPerRow(); i++){
    for(size_t j = 0; j<B.valuesPerCol(); j++){
      EXPECT_NEAR(B.getVal(i,j),mathematica_soln[i][j],1e-5);
    }
  }
  }



}

TEST(SparseTests, choleskyInvert){
  CudaBlockMat<float> L;
  L.setBlockDim(2);
  {
    CudaMat<float> testMat(2,2);
    testMat(0,0)=1    ;testMat(0,1)=0;
    testMat(1,0)=2    ;testMat(1,1)=3;
    testMat.host2dev();
    L.pushBackHost(testMat,0,0);
  }
  {
    CudaMat<float> testMat(2,2);
    testMat(0,0)=4    ;testMat(0,1)=5;
    testMat(1,0)=6    ;testMat(1,1)=7;
    testMat.host2dev();
    L.pushBackHost(testMat,1,0);
  }
  {
    CudaMat<float> testMat(2,2);
    testMat(0,0)=8    ;testMat(0,1)=0;
    testMat(1,0)=9    ;testMat(1,1)=10;
    testMat.host2dev();
    L.pushBackHost(testMat,1,1);
  }
  {
    CudaMat<float> testMat(2,2);
    testMat(0,0)=12    ;testMat(0,1)=13;
    testMat(1,0)=14    ;testMat(1,1)=15;
    testMat.host2dev();
    L.pushBackHost(testMat,2,1);
  }
  {
    CudaMat<float> testMat(2,2);
    testMat(0,0)=16    ;testMat(0,1)=0;
    testMat(1,0)=17    ;testMat(1,1)=18;
    testMat.host2dev();
    L.pushBackHost(testMat,2,2);
  }
  std::cout << std::endl << "L" << std::endl;
  L.printHostValues();

  CudaBlockMat<float> inv = ss::blockMatIdentity(L.rows(),L.getBlockParam().rows);

  ss::choleskySolveMat(L,inv);
  std::cout << std::endl << "inv" << std::endl;
  inv.blocks2host();
  inv.printHostValues();

  {
  std::cout << std::endl << "Comparing ss::choleskySolveMat(L,inv); to the mathematica solution..." << std::endl;
  std::vector<std::vector<float>> mathematica_soln =
  {{1.46696, -0.180755, -0.00412386, -0.0148259, 0.00629973,
    0.000535301}, {-0.180755, 0.194359, -0.0213609, -0.0204198,
    0.0110606, 0.00097174}, {-0.00412386, -0.0213609,
    0.0282883, -0.0110513, -0.0000721873, -0.0000699267}, {-0.0148259,
  -0.0204198, -0.0110513,
    0.0166451, -0.00468871, -0.000366512}, {0.00629973,
    0.0110606, -0.0000721873, -0.00468871,
    0.00739053, -0.00327932}, {0.000535301,
    0.00097174, -0.0000699267, -0.000366512, -0.00327932, 0.00308642}};
  for(size_t i = 0; i<inv.valuesPerRow(); i++){
    for(size_t j = 0; j<inv.valuesPerCol(); j++){
      EXPECT_NEAR(inv.getVal(i,j),mathematica_soln[i][j],1e-5);
    }
  }
  }



}

TEST(SparseTests, Trace){
  CudaBlockMat<float> id = ss::blockMatIdentity(2,2);
  id.blocks2host();
  id.printHostValues();
  EXPECT_NEAR(ss::trace(id.getBlock(0,0)),2.0f,1e-5);
  float trace = ss::trace(id);
  EXPECT_NEAR(trace,4.0f,1e-5);
}

TEST(SparseTests, sparsematOps){
  CudaBlockMat<float> block_mat;
  block_mat.setBlockDim(2);
  {
    CudaMat<float> testMat(2,2);
    testMat(0,0)=1    ;testMat(0,1)=2;
    testMat(1,0)=3    ;testMat(1,1)=4;
    testMat.host2dev();
    block_mat.pushBackHost(testMat,0,0);
  }
  {
    CudaMat<float> testMat(2,2);
    testMat(0,0)=5    ;testMat(0,1)=6;
    testMat(1,0)=7    ;testMat(1,1)=8;
    testMat.host2dev();
    block_mat.pushBackHost(testMat,1,0);
  }
  {
    CudaMat<float> testMat(2,2);
    testMat(0,0)=9    ;testMat(0,1)=10;
    testMat(1,0)=11   ;testMat(1,1)=12;
    testMat.host2dev();
    block_mat.pushBackHost(testMat,1,1);
  }

  CudaBlockMat<float> product = ss::mult(block_mat,block_mat,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_TRANSPOSE);
  std::cout << std::endl << "block_mat" << std::endl;
  block_mat.printHostValues();

  std::cout << std::endl << "csc_block_mat" << std::endl;
  CudaBlockMatCSC<float> csc_block_mat;
  csc_block_mat=block_mat;
  csc_block_mat.printHostValues();

  {
    CudaMat<float> testMat(2,2);
    testMat(0,0)=13   ;testMat(0,1)=14;
    testMat(1,0)=15   ;testMat(1,1)=16;
    testMat.host2dev();
    csc_block_mat.pushBackHost(testMat,1,2);
  }

  std::cout << std::endl << "csc_block_mat with added column" << std::endl;
  csc_block_mat.printHostValues();

  std::cout << std::endl << "csc_block_mat.toCSR" << std::endl;
  csc_block_mat.toCSR().printHostValues();

  std::cout << std::endl << "block_mat.transpose" << std::endl;
  CudaBlockMat<float> transpose = block_mat.transposeHost();
  transpose.printHostValues();


  std::cout << std::endl << "block_mat x block_mat" << std::endl;
  product.blocks2host();
  product.printHostValues();

  ss::add(product,block_mat);
  std::cout << std::endl;
  product.blocks2host();
  product.printHostValues();

  CudaBlockMat<float> block_vector;
  denseParam_t vect_params(2,1);
  block_vector.setBlockParam(vect_params);
  {
    CudaMat<float> testMat(2,1);
    testMat(0,0)=1    ;
    testMat(1,0)=2    ;
    testMat.host2dev();
    block_vector.pushBackHost(testMat,0,0);
  }
  {
    CudaMat<float> testMat(2,1);
    testMat(0,0)=3    ;
    testMat(1,0)=4    ;
    testMat.host2dev();
    block_vector.pushBackHost(testMat,2,0);
  }
  block_vector.printHostValues();
  CudaBlockMat<float> vector_product = ss::mult(block_vector,block_vector,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_TRANSPOSE);
  std::cout << std::endl;
  vector_product.blocks2host();
  vector_product.printHostValues();
  std::cout << "nnz: " << vector_product.nnz() << std::endl;
}
TEST(SparseTests, cholsolve)
{
  // y = m x
  // x = m^-1 y
  // m = L*Transpose(L)
  CudaBlockMat<float> L;
  L.setBlockDim(3);
  {
    CudaMat<float> testMat(3,3);
    testMat(0,0)=1.3f;
    testMat(1,0)=18.2f;testMat(1,1)=2.8f;
    testMat(2,0)=14.3f;testMat(2,1)=0.01f;testMat(2,2)=0.0098f;
    testMat.host2dev();
    L.pushBackHost(testMat,0,0);
  }
  {
    CudaMat<float> testMat1(3,3);
    testMat1(0,2)=1;
    testMat1.host2dev();
    L.pushBackHost(testMat1,1,0);
  }
  {
    CudaMat<float> testMat2(3,3);
    testMat2(0,0)=2;
    testMat2(1,0)=0;testMat2(1,1)=3;
    testMat2(2,0)=0;testMat2(2,1)=0;testMat2(2,2)=4;
    testMat2.host2dev();
    L.pushBackHost(testMat2,1,1);
  }
  CudaBlockMat<float> y;
  {
    CudaMat<float> y1(3,1);
    y.setBlockParam(y1.getDenseParam());
    for(size_t i = 0;i<3; i++){
      y1(i,0)=i;
    }
    y1.host2dev();
    y.pushBackHost(y1,0,0);
  }
  {
    CudaMat<float> y2(3,1);
    for(size_t i = 3;i<6; i++){
      y2(i-3,0)=i;
    }
    y2.host2dev();
    y.pushBackHost(y2,1,0);
  }

  L.getBlock(0,0).printHost();
  L.getBlock(1,0).printHost();
  L.getBlock(1,1).printHost();

  std::cout << "Y: " << std::endl;
  y.getBlock(0,0).printHost();
  y.getBlock(1,0).printHost();

  ss::choleskySolve(L,y);
  std::cout << "Y: " << std::endl;
  y.getBlock(0,0).dev2host();
  y.getBlock(0,0).printHost();
  y.getBlock(1,0).dev2host();
  y.getBlock(1,0).printHost();


}


// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
  testing::InitGoogleTest(&argc, argv);
  ros::init(argc, argv, "sparse_tests");
  ros::NodeHandle nh;
  return RUN_ALL_TESTS();
}
