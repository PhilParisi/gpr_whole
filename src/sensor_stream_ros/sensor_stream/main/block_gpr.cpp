#include "../include/sensor_stream/blockgpr.h"

int main()
{
    BlockGpr testGP;
    //size_t div=100;
    size_t samples=4000;
    size_t itterations = 30;
    float upper,lower;
    upper=10;
    lower=0;
    CudaMat<float> hp(1,1);
    hp(0,0)=10;  // set the hyperparameters
    hp.host2dev();
    testGP.setHyperParam(hp);
    //testGP.genPredVect2d(lower,upper,div);


    CudaMat<float> xall(samples*itterations,2);
    for(size_t n = 0 ; n<itterations ; n++){
        CudaMat<float> x(samples,2); // samples rows, 2 colums to represent 2d point
        for(size_t i = 0 ; i<samples ; i++){
            float rand1 = (std::rand()/((RAND_MAX)/10.0))+n*5;
            float rand2 = std::rand()/((RAND_MAX)/10.0);
            x(i,0) = rand1; // x point
            xall(i+samples*n,0) =rand1;
            x(i,1) = rand2; // y point
            xall(i+samples*n,1) =rand2;
        }
        std::cout<< n << std::endl;
        x.host2dev();
        testGP.addTrainingData(x,x);
    }

//    xall.host2dev();
//    CudaMat<float> groundTruth = testGP.genCovariance(xall,xall);
//    ss::choleskyDecompose(groundTruth);
//    groundTruth.dev2host();
//    groundTruth.printHost();

//    testGP._xTrain.printHost();
//    for(size_t n = 0 ; n<testGP._xTrain.rows() ; n++){
//        testGP._xTrain.getBlock(n).dev2host();
//        testGP._xTrain.getBlock(n).printHost();
//    }
}
