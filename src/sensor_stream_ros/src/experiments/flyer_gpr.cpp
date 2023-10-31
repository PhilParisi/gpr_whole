#include "../../include/sensor_stream_ros/gpr_bag_mapper.h"
#include "../../include/sensor_stream_ros/csv.h"


int main(int argc, char **argv)
{


//  std::shared_ptr<BlockGpr> gpr;
//  gpr.reset(new BlockGpr);
//  gpr->params.regression.reset(new gpr::RegressionParams);
//  gpr->params.prediction.reset(new gpr::PredParams);
//  gpr->params.regression->kernel=&gpr::kernels::sqExpKernel2d;
//  gpr->params.regression->hyperParam.reset(1,1);
//  gpr->params.regression->hyperParam.initHost();
//  gpr->params.regression->hyperParam(0,0)=100.0f;
//  gpr->params.regression->hyperParam.host2dev();
//  gpr->params.regression->sigma2e = 0.4f;
//  gpr->params.regression->nnzThresh=1e-30f;
//  int block_size = 400;


//  io::CSVReader<3> in("/home/kris/Data/flyer_gpr/o2_gpr.csv");
//  in.read_header(io::ignore_extra_column, "value", "depth", "along_track");
//  float value; float depth; float along_track;
//  while(in.read_row(value, depth, along_track)){
//    CudaMat<float> x(block_size,2);
//    CudaMat<float> y(block_size,1);
//    for(size_t i=0; i<block_size ; i++){
//        in.read_row(value, depth, along_track);
//        x(i,0)=along_track/100.0f;
//        //std::cout<<along_track;
//        x(i,1)=-depth;
//        y(i,0)=value-179.6f;

//    }
//    //x.printHost();
//    x.host2dev();
//    y.host2dev();
//    gpr->addTrainingData(x,y);

//  }
//  cv::Mat sparsity = ss::viz::sparsity2image(gpr->getCholeskyFactor(),2.0f*float(block_size)/float(100));
//  cv::imshow( "SparsityPattern",
//              sparsity
//              );
//  cv::waitKey(0);


//  gpr->predict(2555.0f/100.0f,3555.0f/100.0f,0,-300,200,100);


//  pcl::PointCloud<pcl::PointXYZI>::Ptr _predictedCloud;
//  _predictedCloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
//  _predictedCloud->width  = gpr->prediction.points.rows();
//  _predictedCloud->height  = 1;
//  _predictedCloud->points.resize(gpr->prediction.points.rows());
//  gpr->prediction.mu.dev2host();
//  gpr->prediction.sigma.dev2host();
//  gpr->prediction.points.dev2host();

//  for(size_t i = 0 ; i<_predictedCloud->size() ; i++){
//      _predictedCloud->points[i].x=gpr->prediction.points(i,0)*100.0f;
//      _predictedCloud->points[i].y=gpr->prediction.points(i,1);
//      if(isnan(gpr->prediction.mu(i,0)))
//          _predictedCloud->points[i].z=0;
//      else
//          _predictedCloud->points[i].z=gpr->prediction.mu(i,0);
//      _predictedCloud->points[i].intensity=gpr->prediction.sigma(i,0);
//  }

//  std::cout<< "saving... "<< std::endl;
//  pcl::io::savePCDFileASCII("/home/kris/Data/flyer_gpr/o2_gpr.pcd",*_predictedCloud);
}
