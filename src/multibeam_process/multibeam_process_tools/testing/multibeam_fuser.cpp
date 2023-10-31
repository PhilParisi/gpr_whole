//#include <iostream>
//#include <navfuser.h>
//#include <navestparser.h>
//#include <stdio.h>
//#include "../include/multibeam_process/km_datagram.h"
//#include <octomap/octomap.h>
//#include <octomap/OcTree.h>

//long getFileSize(FILE *file)
//{
//  long lCurPos, lEndPos;
//  lCurPos = ftell(file);
//  fseek(file, 0, 2);
//  lEndPos = ftell(file);
//  fseek(file, lCurPos, 0);
//  return lEndPos;
//}

//int main(int argc, char *argv[])
//{

//  ros::init(argc, argv, "multibeam_fuser");
//  NavFuser nav;

//  octomap::OcTree tree (0.1);

//  std::shared_ptr<NavEstParser> parser;
//  parser.reset(new NavEstParser);
//  parser->readDVZ("/home/kris/Data/cave_data/NAV/dvz.csv");
//  parser->readParo("/home/kris/Data/cave_data/NAV/ctd.csv");

//  nav.setNavParser(parser);

//  ///
//  /// read file stuff
//  ///

//  const char *filePath = "/home/kris/Data/cave_data/2017,Jul,19,03-34-59.all";
//  BYTE *fileBuf;			// Pointer to our buffered data
//  FILE *file = NULL;		// File pointer

//  // Open the file in binary mode using the "rb" format string
//  // This also checks if the file exists and/or can be opened for reading correctly
//  if ((file = fopen(filePath, "rb")) == NULL)
//    std::cout << "Could not open specified file" << std::endl;
//  else
//    std::cout << "File opened successfully" << std::endl;

//  // Get the size of the file in bytes
//  long fileSize = getFileSize(file);

//  // Allocate space in the buffer for the whole file
//  fileBuf = new BYTE[fileSize];

//  // Read the file in to the buffer
//  fread(fileBuf, fileSize, 1, file);

//  KmDatagram data;
//  unsigned long offset=0;
//  data.readFromByteArray(fileBuf,offset);


//  int i=0;
//  while(offset<=fileSize){
//    data.readFromByteArray(fileBuf,offset);

////    if (data.header.time>13012901)
////      break;
//      if(data.header.type=='X'){
//        std::shared_ptr<KmXyzData> xyzPtr;
//        xyzPtr = data.xyzData;

//        pcl::PointCloud<pcl::PointXYZ> pt;
//        pt = nav.projectToMapframe(data.header.getDoubleTime(),xyzPtr->x,xyzPtr->y,xyzPtr->z,xyzPtr->detectInfo);
//        octomap::Pointcloud cloud;

//        geometry_msgs::PointStamped origin;
//        origin = nav.projectToMapframe(data.header.getDoubleTime(),0,0,0);
//        octomap::point3d sensorOrigin(origin.point.x,origin.point.y,origin.point.z);

//        for (pcl::PointCloud<pcl::PointXYZ>::const_iterator it = pt.begin(); it != pt.end(); ++it){
//          cloud.push_back(it->x, it->y, it->z);
//        }
//        tree.insertPointCloud(cloud,sensorOrigin);
//        //ros::Duration(0.08).sleep();
//      }
//      //data.header.getDoubleTime();
//    i++;
//    //std::cout<<i<<std::endl;
//  }
//  tree.writeBinary("simple_tree.bt");
//  return 0;
//}
