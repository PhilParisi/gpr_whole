#include "../../include/sensor_stream/blockgpr.h"

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/impl/pcd_io.hpp>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/filters/passthrough.h>

int main(int argc, char** argv)
{

    // Load point clouds
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZI>);
    if (pcl::io::loadPCDFile<pcl::PointXYZI> (argv[1], *cloud) == -1) //* load the file
    {
        std::cerr<< "Couldn't read file " << argv[1] << std::endl;
        return (-1);
    }
    std::cout << "the following file was loaded: " << argv[1] << std::endl;
    std::cout << "it has size: " << cloud->size() << std::endl;


    pcl::PointXYZI minPt, maxPt;
    pcl::getMinMax3D (*cloud, minPt, maxPt);
    std::cout << "Max x: " << maxPt.x << std::endl;
    std::cout << "Max y: " << maxPt.y << std::endl;
    std::cout << "Max z: " << maxPt.z << std::endl;
    std::cout << "Min x: " << minPt.x << std::endl;
    std::cout << "Min y: " << minPt.y << std::endl;
    std::cout << "Min z: " << minPt.z << std::endl;


    //---------------
    // setup GP
    //---------------



    //---------------
    // add training data
    //---------------
    float tileDim=10; //meters
    float trainingDim=1.5 ;
    size_t outIndex=0;

    for(float tileMinX=minPt.x ; tileMinX<maxPt.x; tileMinX+=tileDim){
        for(float tileMinY=minPt.y ; tileMinY<maxPt.y; tileMinY+=tileDim){
            try{
            BlockGpr testGP;
            //testGP.sigma2e = 1;
            testGP.sigma2e = .2;//
            size_t samples = cloud->size();
            //size_t samples = 8;
            CudaMat<float> hp(2,1);
            //hp(0,0)=.005;  // set the hyperparameters
            //hp(0,0)=40;  // seamount length scale
            hp(0,0)=.8;  //  length scale
            hp(1,0)=1;  // seamount alpha

            hp.host2dev();
            testGP.setHyperParam(hp);
            testGP.setKernel(&sqExpKernel2d);


            size_t blockSize=240;  //integer
            float xMin,xMax,yMin,yMax;
            xMin=tileMinX-trainingDim;
            xMax=tileMinX+tileDim+trainingDim;
            yMin=tileMinY-trainingDim;
            yMax=tileMinY+tileDim+trainingDim;

            for(size_t pointIndex=0;pointIndex < samples; pointIndex++){
                size_t blockIndex=0;
                CudaMat<float> x(blockSize,2);
                CudaMat<float> train(blockSize,1);
                while(pointIndex<samples && blockIndex<blockSize ){
                    if(cloud->points[pointIndex].x > xMin && cloud->points[pointIndex].x < xMax &&
                       cloud->points[pointIndex].y > yMin && cloud->points[pointIndex].y < yMax){
                        x(blockIndex,0) = cloud->points[pointIndex].x; // x point
                        x(blockIndex,1) = cloud->points[pointIndex].y; // y point
                        train(blockIndex,0) = cloud->points[pointIndex].z; // z point
                        blockIndex++;
                    }
                    pointIndex++;
                }
                //if blockIndex<
                x.host2dev();
                train.host2dev();
                testGP.addTrainingData(x,train);
                printf("Tile no. %i \n",outIndex);
            }



            //testGP.solve();
            size_t divx,divy;
            divx=40;
            divy=40;
            //testGP.predict(0,.6,0,.4,divx,divy);
            testGP.predict(tileMinX,tileMinX+tileDim,tileMinY,tileMinY+tileDim,divx,divy);

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
            pcl::io::savePCDFileASCII("/home/kris/Data/SensorStream/bluff_survey/tiles"+ std::to_string(outIndex)+".pcd",*outCloud);
            }
            catch (...) {
              std::cout << "Exception occurred";
            }

            outIndex++;

        }
    }
    return 0;

}
