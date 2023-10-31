#include "../../include/sensor_stream/blockgpr.h"

void createTest(){
  CudaMat<float> test(10000,10000,host);
  test.host2dev();
  test.dev2host();
}

void GprTest(){
  std::cout << "  num cudamats start: " << CudaMat<float>::getNumObjects() << std::endl;
  Gpr testGP;
  testGP.genPredVect2d(0,40,40);

  CudaMat<float> hp(1,1);
  hp(0,0)=10;  // set the hyperparameters
  hp.host2dev();
  testGP.setHyperParam(hp);

  size_t numPoints = 1000;
  CudaMat<float> x(numPoints,2);
  for(size_t i = 0 ; i<numPoints ; i++){
      x(i,0) = std::rand()/((RAND_MAX)/10.0);
      x(i,1) = std::rand()/((RAND_MAX)/10.0);
  }

  CudaMat<float> y(numPoints,1);
  for(size_t i = 0; i<numPoints; i++){
      y(i,0) = sin(x(i,0))+cos(x(i,1));
  }
  x.host2dev();
  y.host2dev();

  testGP.setTrainingData(x,y);

  //testGP.solve();
  std::cout << "  num cudamats end: " << CudaMat<float>::getNumObjects() << std::endl;
}



int main(int argc, char** argv){

  for(size_t i = 0; i<10000; i++){
    std::cout << "itteration: " << i << std::endl;
    //createTest();
    GprTest();
  }

  return 0;
}
