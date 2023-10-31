#ifndef CUDASPARSEMAT_H
#define CUDASPARSEMAT_H

#include <memory>  // std::shared_ptr
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <cusparse_v2.h>
#include "cudamat.h"
#include "abstractcsrmat.h"
//#include "sparsemat_operations.h"

template <class T>
class CudaSparseMat: public AbstractCsrMat{
public:
    CudaSparseMat();
    void reset(size_t dataSize=0);
    void initFromDense(CudaMat<float> in);
    void initDev();
    void initHost();
    void dev2host();
    void host2dev();

    void pushBackHost(T val, unsigned int i, unsigned int j);

    void printHost();
    void printHostData(int i);
    void printZero();
    void printHostRawData();
    void printHostCsrRow();
    void printHostCsrCol();

    int rows(cusparseOperation_t op =  CUSPARSE_OPERATION_NON_TRANSPOSE) const{
      if(op==CUSPARSE_OPERATION_TRANSPOSE||op==CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE)
        return _cols;
      else
        return _rows;
    }
    int cols(cusparseOperation_t op =  CUSPARSE_OPERATION_NON_TRANSPOSE) const{
      if(op==CUSPARSE_OPERATION_TRANSPOSE||op==CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE)
        return _rows;
      else
        return _cols;
    }

    T val(size_t i, size_t j);

    T * rawValPtr();
    int * rawCsrRowPtr();
    int * rawCsrColPtr();
    size_t nnz();

    void addRows(CudaSparseMat<float> newRows, cusparseOperation_t transNewRows = CUSPARSE_OPERATION_NON_TRANSPOSE);
    void transposeDev();
    cusparseMatDescr_t getDescriptor();
    //cusparseHandle_t handle;

protected:
    std::shared_ptr<thrust::device_vector<int> > d_csrRowPtr;
    std::shared_ptr<thrust::device_vector<int> > d_csrColInd;
    std::shared_ptr<thrust::device_vector<T> > d_csrVal;

    std::shared_ptr<thrust::host_vector<int> > h_csrRowPtr;
    std::shared_ptr<thrust::host_vector<int> > h_csrColInd;
    std::shared_ptr<thrust::host_vector<T> > h_csrVal;

    cusparseMatrixType_t _matType;
    cusparseIndexBase_t _baseType;

    size_t _dataSize;
    int _rows;
    int _cols;
    int _nnz; // number non zero
};




#endif // CUDASPARSEMAT_H
