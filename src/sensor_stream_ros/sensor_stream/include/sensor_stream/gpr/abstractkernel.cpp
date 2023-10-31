#include "abstractkernel.h"

namespace ss { namespace kernels {

AbstractKernel::AbstractKernel()
{

}

float & AbstractKernel::hyperparam(std::string key){
  return hyperparams_(hp_indices_[key]);
}

float & AbstractKernel::hyperparam(size_t index){
  return hyperparams_(index);
}
}}
