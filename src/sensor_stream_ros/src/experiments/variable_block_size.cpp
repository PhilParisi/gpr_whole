#include "../../include/sensor_stream_ros/gpr_bag_mapper.h"


int main(int argc, char **argv)
{
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
//  {
//    GprBagMapper mapper(bag);
//    mapper.analytics.directory=output_dir;
//    mapper.analytics.test_name="block_size_0032";
//    mapper.mapper.cfg.block->size=32;
//    mapper.run();
//  }
//  {
//    GprBagMapper mapper(bag);
//    mapper.analytics.directory=output_dir;
//    mapper.analytics.test_name="block_size_0100";
//    mapper.mapper.cfg.block->size=100;
//    mapper.run();
//  }
//  {
//    GprBagMapper mapper(bag);
//    mapper.analytics.directory=output_dir;
//    mapper.analytics.test_name="block_size_0200";
//    mapper.mapper.cfg.block->size=200;
//    mapper.run();
//  }
//  {
//    GprBagMapper mapper(bag);
//    mapper.analytics.directory=output_dir;
//    mapper.analytics.test_name="block_size_0400";
//    mapper.mapper.cfg.block->size=400;
//    mapper.run();
//  }
//  {
//    GprBagMapper mapper(bag);
//    mapper.analytics.directory=output_dir;
//    mapper.analytics.test_name="block_size_0032";
//    mapper.mapper.cfg.block->size=32;
//    mapper.run();
//  }
//  {
//    GprBagMapper mapper(bag);
//    mapper.analytics.directory=output_dir;
//    mapper.analytics.test_name="block_size_0100";
//    mapper.mapper.cfg.block->size=100;
//    mapper.run();
//  }
  {
    GprBagMapper mapper(bag);
    mapper.analytics.directory=output_dir;
    mapper.analytics.test_name="block_size_0800";
    mapper.mapper.cfg.block->size=800;
    mapper.run();
  }
//  {
//    GprBagMapper mapper(bag);
//    mapper.analytics.directory=output_dir;
//    mapper.analytics.test_name="block_size_1600";
//    mapper.mapper.cfg.block->size=1600;
//    mapper.run();
//  }
//  {
//    GprBagMapper mapper(bag);
//    mapper.analytics.directory=output_dir;
//    mapper.analytics.test_name="block_size_2400";
//    mapper.mapper.cfg.block->size=2400;
//    mapper.run();
//  }
}
