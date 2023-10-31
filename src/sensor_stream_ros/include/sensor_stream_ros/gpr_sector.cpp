#include "gpr_sector.h"
namespace ss{namespace ros {
  GprSector::GprSector(){
    _params.reset(new GprSectorParams);
  }

  std::vector<SingleFrameBlock> GprSector::getBlocks(){
    _tiler->getTileParam()->xdim;

  }

}}
