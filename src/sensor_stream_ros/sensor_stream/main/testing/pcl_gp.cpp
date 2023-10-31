#include "../../include/sensor_stream/blockgpr.h"

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/impl/pcd_io.hpp>
#include <pcl/point_types.h>

int main(int argc, char** argv)
{

    // Load point clouds
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ> (argv[1], *cloud) == -1) //* load the file
    {
        std::cerr<< "Couldn't read file " << argv[1] << std::endl;
        return (-1);
    }
    std::cout << "the following file was loaded: " << argv[1] << std::endl;
    std::cout << "it has size: " << cloud->size() << std::endl;



    //---------------
    // setup GP
    //---------------
    BlockGpr testGP;
    size_t samples = cloud->size();
    //size_t samples = 8;
    CudaMat<float> hp(1,1);
    hp(0,0)=4000;  // set the hyperparameters
    hp.host2dev();
    testGP.setHyperParam(hp);
    testGP.setKernel(&sqExpKernel2d);

//    CudaMat<float> testPredBlock = testGP.genPredBlock(-2,2,1,4,5,4);
//    testPredBlock.dev2host();
//    testPredBlock.printHost();


    //---------------
    // add training data
    //---------------
    size_t divisions = 20;
    for(size_t batch=0; batch<divisions ; batch++){
        CudaMat<float> x(samples/divisions,2);
        CudaMat<float> train(samples/divisions,1);
        for(size_t index=0 ; index<samples/divisions ; index++){  // for every point in the cloud
            size_t i=samples/divisions*batch+index;
            x(index,0) = cloud->points[i].x; // x point
            x(index,1) = cloud->points[i].y; // y point
            train(index,0) = cloud->points[i].z; // z point

        }
        x.host2dev();
        train.host2dev();
        testGP.addTrainingData(x,train);
        printf("itteration %i",batch);
        //x.printHost();

    }



    //testGP.solve();
    size_t divx,divy;
    divx=200;
    divy=400;
    //testGP.predict(0,.6,0,.4,divx,divy);
    testGP.predict(284,1500,-1000,850,divx,divy);

    //--------------
    // Create output cloud
    //--------------
    pcl::PointCloud<pcl::PointXYZI>::Ptr outCloud (new pcl::PointCloud<pcl::PointXYZI>);
    outCloud->width  = divx * divy;
    outCloud->height  = 1;
    outCloud->points.resize (divx * divy);
    testGP._pred.dev2host();
    testGP._mu.dev2host();
    testGP._var.dev2host();
    //testGP._mu.printHost();
    for(size_t i = 0 ; i<outCloud->size() ; i++){
        outCloud->points[i].x=testGP._pred(i,0);
        outCloud->points[i].y=testGP._pred(i,1);
        if(std::isnan(testGP._mu(i,0)))
            outCloud->points[i].z=0;
        else
            outCloud->points[i].z=testGP._mu(i,0);
        //outCloud->points[i].intensity=testGP._mu(i,0);
        outCloud->points[i].intensity=testGP._var(i,0);
    }
    std::cout<< "saving... "<< std::endl;
    pcl::io::savePCDFileASCII("/home/kris/Data/SensorStream/seamount8/NA101_Seamount8_160k_GP.pcd",*outCloud);


}
