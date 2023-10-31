#ifndef CUDABLOCKMATCSC_H
#define CUDABLOCKMATCSC_H
#include "cudablockmat.h"


template <class T>
/*!
 * \brief This class is simply an interface to represent a CudaBlockMat as CSC (compressed sparse column)
 */
class CudaBlockMatCSC : public CudaBlockMat<T>
{
public:
  CudaBlockMatCSC(){}

  void pushBackHost(CudaMat<T> val, size_t i, size_t j){
    CudaBlockMat<T>::pushBackHost(val,j,i);
  }

  CudaMat<T> & getBlock(size_t row, size_t col, cusparseOperation_t op =  CUSPARSE_OPERATION_NON_TRANSPOSE){
    return CudaBlockMat<T>::getBlock(col,row,op);
  }

  CudaBlockMat<T> toCSR(){
    return this->transposeHost();
  }

  int rows(cusparseOperation_t op =  CUSPARSE_OPERATION_NON_TRANSPOSE) const{
    AbstractCsrMat::cols(op);
  }

  int cols(cusparseOperation_t op =  CUSPARSE_OPERATION_NON_TRANSPOSE) const{
    AbstractCsrMat::rows(op);
  }

  CudaBlockMatCSC<T> operator=(CudaBlockMat<T>& other){
    CudaBlockMat<T> * block_mat = this;
    *block_mat = other.transposeHost();
    return *this;
  }


};

#endif // CUDABLOCKMATCSC_H
