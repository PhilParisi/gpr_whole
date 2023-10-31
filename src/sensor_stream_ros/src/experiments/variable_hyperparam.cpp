#include "../../include/sensor_stream_ros/gpr_bag_mapper.h"


int main(int argc, char **argv)
{
  /*
  ros::init(argc, argv, "variable_block_size");
  std::shared_ptr<rosbag::Bag> bag;
  bag.reset(new rosbag::Bag);
  if(argc < 2){
    std::cerr << "ERROR: You must input a bag file" << std::endl;
    return 1;
  }
  if(argc < 3){
    std::cerr << "ERROR: You must input an output directory" << std::endl;
    return 1;
  }
  bag->open(argv[1]);
  std::string output_dir=argv[2];

  {
    GprBagMapper mapper(bag);
    mapper.analytics.directory=output_dir;
    mapper.analytics.test_name="block_size_0400";
    mapper.mapper.cfg.block->size=400;
    mapper.mapper.cfg.tile->xMargin = 10;
    mapper.mapper.cfg.tile->yMargin = 10;
    mapper.processBag();

    HyperparamAnalytics analytics;
    analytics.test_name="sigma_length_slide";
    analytics.directory = output_dir;
    for(float i = -3; i <= 0 ; i+=0.05f){
      for(float j = -1; j <= 2 ; j+=0.05f){
        float sigma = pow(10,i);
        float length = pow(10,j);
//        sigma = 0.01;
//        length = 3.16228;
        mapper.mapper.cfg.gpr.regression->hyperParam(0,0)=length;
        mapper.mapper.cfg.gpr.regression->hyperParam.host2dev();
        mapper.mapper.cfg.gpr.regression->sigma2e=sigma;
        mapper.mapper.doTile(mapper.mapper.getQueue().operator[](12));
        std::cout<< sigma << "," << length << "," << mapper.mapper.gpr->prediction.LML << std::endl;
        cv::Mat sparsity = ss::viz::sparsity2image(mapper.mapper.gpr->getCholeskyFactor(),2.0f*float(mapper.mapper.cfg.block->size)/float(100));
        cv::imshow( "SparsityPattern",
                    sparsity
                    );
        cv::waitKey(100);
        analytics.sigma2e.push_back(sigma);
        std::vector<float> hpVec;
        for(size_t i = 0; i< mapper.mapper.cfg.gpr.regression->hyperParam.size();i++){
          hpVec.push_back(mapper.mapper.cfg.gpr.regression->hyperParam(i,0) );
        }
        analytics.hyperparams.push_back(hpVec);
        analytics.lml.push_back(mapper.mapper.gpr->prediction.LML);
        analytics.write();

      }
    }
  }
*/
//  {
//    GprBagMapper mapper(bag);
//    mapper.processBag();
//    mapper.analytics.directory=output_dir;
//    mapper.analytics.test_name="block_size_0400";
//    mapper.mapper.cfg.block->size=400;

//  for(float sigma = 0.5f; sigma <= 4.0f ; sigma+=0.5f){
//    for(float length = 0.5f; length < 3.0f ; length+=0.5f){
//      std::cout<< "length: " << length << std::endl;
//      std::cout<< "sigma: " << sigma << std::endl;
//      mapper.mapper.cfg.gpr.regression->hyperParam(0,0)=length;
//      mapper.mapper.cfg.gpr.regression->hyperParam.host2dev();
//      mapper.mapper.cfg.gpr.regression->sigma2e=sigma;
//      //mapper.mapper.doTile(mapper.mapper.getQueue().operator[](12));
//      mapper.run();
//      std::cout<< sigma << "," << length << "," << mapper.mapper.gpr->prediction.LML << std::endl;
//      cv::Mat sparsity = ss::viz::sparsity2image(mapper.mapper.gpr->getCholeskyFactor(),2.0f*float(mapper.mapper.cfg.block->size)/float(100));
//      cv::imshow( "SparsityPattern",
//                  sparsity
//                  );
//      cv::waitKey(100);
//    }
//  }
//    //mapper.run();
//  }

}
