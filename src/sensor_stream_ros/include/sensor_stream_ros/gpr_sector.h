#ifndef GPR_SECTOR_H
#define GPR_SECTOR_H
#include <vector>
#include <memory>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <sensor_stream/blockgpr.h>

#include "block_tiler.h"
namespace ss{namespace ros {

  struct GprSectorParams{
    pcl::PointXYZ center;
    float xPredDim;
    float yPredDim;
    float xTrainingMargin;
    float yTrainingMargin;
  };
  typedef std::shared_ptr<GprSectorParams> GprSectorParamsPtr;


  /*!
   * \brief describes a working area of a gaussian process regression.
   * a working area is defined as a prediction area and some margin beyond the training area
   * for training data
   */
  class GprSector
  {
  public:
    GprSector();
    std::vector<SingleFrameBlock> getBlocks();
  private:
    GprSectorParamsPtr _params;
    BlockTilerPtr _tiler;
  };

}}

#endif // GPR_SECTOR_H
