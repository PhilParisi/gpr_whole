#ifndef FIXEDHPKERNEL_H
#define FIXEDHPKERNEL_H

#include "abstractkernel.h"

namespace ss { namespace kernels {

template <int n_hp>
class FixedHPKernel : public AbstractKernel
{
public:
  static const int    num_hp = n_hp;
};

}}
#endif // FIXEDHPKERNEL_H
