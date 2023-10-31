#include "bpslam_particle_data.h"

namespace ss{ namespace bpslam {

bool PredictionReigon::doesOverlap(PredictionReigon & other, float margin){
  return   other.min_point.x < max_point.x+margin
        && other.max_point.x > min_point.x-margin
        && other.min_point.y < max_point.y+margin
        && other.max_point.y > min_point.y-margin;
}

bool PredictionReigon::doesOverlap(SingleFrameBlock::Ptr block, float margin){
  return   block->getCenterOfMass().x < max_point.x+margin
        && block->getCenterOfMass().x > min_point.x-margin
        && block->getCenterOfMass().y < max_point.y+margin
        && block->getCenterOfMass().y > min_point.y-margin;
}

  bool BPSLAMParticleData::doesOverlap(BPSLAMParticleData::Ptr other, float margin){
    return   map.prediction_reigon.doesOverlap(other->map.prediction_reigon);
  }

  bool BPSLAMParticleData::doesOverlap(SingleFrameBlock::Ptr block, float margin){
    return   map.prediction_reigon.doesOverlap(block);
  }

}}
