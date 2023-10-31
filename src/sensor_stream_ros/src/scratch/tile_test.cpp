#include "../include/sensor_stream_ros/block_tiler.h"

int main(int argc, char **argv){
    ss::ros::TileParamsPtr tileParams;
    tileParams.reset(new ss::ros::TileParams);
    tileParams->xdim = 2;
    tileParams->ydim = 2;
    ss::BlockParams::Ptr blockParams;
    blockParams.reset(new ss::BlockParams);
    blockParams->size = 4;


    ss::ros::BlockTiler tiler(tileParams,blockParams);

    pcl::PointCloud<pcl::PointXYZI>::Ptr outCloud (new pcl::PointCloud<pcl::PointXYZI>);
    outCloud->width  = blockParams->size;
    outCloud->height  = 1;
    outCloud->points.resize (blockParams->size);

    outCloud->points[0].x=1.0f;
    outCloud->points[0].y=1.5f;

    outCloud->points[1].x=0.5f;
    outCloud->points[1].y=0.8f;

    outCloud->points[2].x=1.3f;
    outCloud->points[2].y=1.2f;

    outCloud->points[3].x=2.1f;
    outCloud->points[3].y=0.0f;



    ss::SingleFrameBlock testBlock(blockParams);
    testBlock.setPointcloud(outCloud);

    tiler.addBlock(testBlock);

    return 0;
}

