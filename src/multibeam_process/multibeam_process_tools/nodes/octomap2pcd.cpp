#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <octomap/octomap.h>
#include <octomap/OcTree.h>
#include <memory.h>

int main(int argc, char *argv[])
{
  if(argc<3){
    std::cout << "usage: \n"
                 "octomap2pcd <octomap file> <output pcd file>" <<std::endl;
    return 0;
  }

  std::string octomap_file = argv[1];
  std::string pcd_file = argv[2];

  std::shared_ptr<octomap::OcTree> mapPtr_;
  mapPtr_=std::make_shared<octomap::OcTree>(1.0);
  if(!mapPtr_->readBinary(octomap_file)){
      return 1;
  }


  pcl::PointCloud<pcl::PointXYZ> cloud;

  for (octomap::OcTree::iterator it = mapPtr_->begin(),end = mapPtr_->end(); it != end; ++it){
    if (mapPtr_->isNodeOccupied(*it)){
      double x = it.getX();
      double y = it.getY();
      double z = it.getZ();
      double size = it.getSize();
      cloud.push_back(pcl::PointXYZ(x, y, z));
    }
  }
  pcl::io::savePCDFileASCII(pcd_file, cloud);

}
