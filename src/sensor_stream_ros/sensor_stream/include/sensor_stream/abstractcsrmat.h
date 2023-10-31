#ifndef ABSTRACTCSRMAT_H
#define ABSTRACTCSRMAT_H

#include <memory>  // std::shared_ptr
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <cusparse_v2.h>
#include "cudamat.h"
#include "abstractmat.h"


struct CsrParams_t
{
    CsrParams_t(){
        dataSize=0;
        rows=0;
        cols=0;
        nnz=0;
    }
    cusparseMatrixType_t matType;
    cusparseIndexBase_t baseType;

    size_t dataSize;
    int rows;
    int cols;
    int nnz; // number non zero
};

class AbstractCsrMat: public AbstractMat
{
public:
    AbstractCsrMat();
    ~AbstractCsrMat();
    void reset();
    /*!
     * \brief copies device data to host (calls proctected virtual function dev2hostData()
     * to copy child class data)
     */
    void dev2host();
    /*!
     * \brief copies host data to device (calls proctected virtual function host2devData()
     * to copy child class data)
     */
    void host2dev();

    int primaryDim(){
      return _csrParams->rows;
    }
    int secondaryDim(){
      return _csrParams->cols;
    }

    /*!
     * \brief Returns the rows of the matrix you want to represent DO NOT USE TO get the primary diemntion for
     * CSR operations as this function is virutal and can be overridden
     * \param op
     * \return
     */
    virtual int rows(cusparseOperation_t op =  CUSPARSE_OPERATION_NON_TRANSPOSE) const {
      if(op==CUSPARSE_OPERATION_NON_TRANSPOSE)
        return _csrParams->rows;
      else
        return _csrParams->cols;
    }
    /*!
     * \brief Gets the colums of the matrix you want to represent DO NOT USE for the secondary dimention for CSR operations
     * as this function is virutal and can be overridden
     * \param op
     * \return
     */
    virtual int cols(cusparseOperation_t op =  CUSPARSE_OPERATION_NON_TRANSPOSE) const {
      if(op==CUSPARSE_OPERATION_NON_TRANSPOSE)
        return _csrParams->cols;
      else
        return _csrParams->rows;
    }

    int HostCsrRow(size_t i) const {return h_csrRowPtr->operator[](i);}
    int HostCsrCol(size_t i) const {return h_csrColInd->operator[](i);}
    // COO accessors
//    int hostCooCol(size_t i){return h_csrColInd->operator[](i);}
//    int hostCooRow(size_t i){return h_cooRowInd->operator[](i);}
//    void updateCooRow();
//    void cooDev2Host(){*h_cooRowInd=*d_cooRowInd;}

    int * rawCsrRowPtr(){return thrust::raw_pointer_cast(d_csrRowPtr->data());}
    int * rawCsrColPtr(){return thrust::raw_pointer_cast(d_csrColInd->data());}
//    int * rawCooRowPtr(){return thrust::raw_pointer_cast(d_cooRowInd->data());}

    size_t nnz() const {return _csrParams->nnz;}

    void printHost();
    void printHostCsrRow();
    void printHostCsrCol();

    cusparseHandle_t & getHandle(){return handle;}


protected:
    virtual void resetData(){return;}
    /*!
     * \brief dev2hostData should be overridden by the child class to copy value data from device to the host
     */
    virtual void dev2hostData(){std::cerr<<"dev2hostData() was called but not implemented"<<std::endl;}
    /*!
     * \brief host2devData should be overridden by the child class to copy value data from host to the device
     */
    virtual void host2devData(){std::cerr<<"host2devData() was called but not implemented"<<std::endl;}
    /*!
     * \brief a virtual method that should print the host data at index i on one line and equally spaced
     * \param i the csrValue index you want to print
     */
    virtual void printHostData(int i){printf ("i:%4i   ", i);}

    virtual void printZero(){      std::cout<<"zero     ";}
    /*!
     * \brief pushBackCsr should be called after a child class pushBack(<datatype> is called)
     * this function simply increments the CSR parameters.
     * \param i row
     * \param j col
     */
    void pushBackCsr(size_t i, size_t j);
    void pushBackEmpty(size_t i , size_t j);

    int &hostCsrColInd(size_t idx){return h_csrColInd->operator[](idx);}
    int &hostCsrRowPtr(size_t idx){return h_csrRowPtr->operator[](idx);}

    std::shared_ptr<thrust::device_vector<int> > d_csrRowPtr;
    std::shared_ptr<thrust::device_vector<int> > d_csrColInd;
//    std::shared_ptr<thrust::device_vector<int> > d_cooRowInd;


    std::shared_ptr<thrust::host_vector<int> > h_csrRowPtr;
    std::shared_ptr<thrust::host_vector<int> > h_csrColInd;
//    std::shared_ptr<thrust::host_vector<int> > h_cooRowInd;

    std::shared_ptr<CsrParams_t> _csrParams;

    static cusparseHandle_t handle;
};

#define cusparsecall(call)                                                                                        \
    do                                                                                                          \
    {                                                                                                           \
        cusparseStatus_t status = (call);                                                                         \
        if(CUSPARSE_STATUS_SUCCESS != status)                                                                     \
        {                                                                                                       \
            fprintf(stderr,"CUSPARSE Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status);     \
            cudaDeviceReset();                                                                                  \
            exit(EXIT_FAILURE);                                                                                 \
        }                                                                                                       \
                                                                                                                \
    }                                                                                                           \
    while(0)

#endif // ABSTRACTCSRMAT_H
