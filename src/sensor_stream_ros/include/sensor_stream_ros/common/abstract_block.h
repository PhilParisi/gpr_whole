#ifndef ABSTRACTBLOCK_H
#define ABSTRACTBLOCK_H

#include <sensor_stream/include/sensor_stream/cudablockmat.h>
#include <memory>
namespace ss {

struct TrainingBlock{
    CudaMat<float> x;
    CudaMat<float> y;
    CudaMat<float> variance;
    typedef std::shared_ptr<TrainingBlock> Ptr;
};

struct BlockParams{
    size_t size;
    typedef std::shared_ptr<BlockParams> Ptr;
};

class AbstractBlock{
public:
  typedef std::shared_ptr<AbstractBlock> Ptr;
  AbstractBlock();
  void setParams(BlockParams::Ptr params){params_=params;}
  /*!
   * \brief set a shared pointer to the params
   * \param params a shared ptr to the params
   */
  BlockParams::Ptr getParams();
  virtual size_t size()=0;
protected:
  BlockParams::Ptr params_;
};

}

#endif // ABSTRACTBLOCK_H
