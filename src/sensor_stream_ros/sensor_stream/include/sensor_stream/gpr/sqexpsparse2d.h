#ifndef SQEXPSPARSE2D_H
#define SQEXPSPARSE2D_H

#include "fixedhpkernel.h"

namespace ss { namespace kernels {
#define F_PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062f
class SqExpSparse2d : public FixedHPKernel<2>
{
public:
  typedef std::shared_ptr<SqExpSparse2d> Ptr;
  SqExpSparse2d();

  CudaMat<float> genCovariance(CudaMat<float> & x1, CudaMat<float> & x2);
  CudaMat<float> genCovarianceWithSensorNoise(CudaMat<float> & x1, CudaMat<float> obs_var);
  CudaMat<float> partialDerivative(CudaMat<float> & x1, CudaMat<float> & x2, size_t hp_index);

};

}}

#endif // SQEXPSPARSE2D_H
