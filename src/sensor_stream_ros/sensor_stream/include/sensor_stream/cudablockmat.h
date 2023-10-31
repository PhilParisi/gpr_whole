#ifndef CUDABLOCKMAT_H
#define CUDABLOCKMAT_H

#include "abstractcsrmat.h"

template <class T>
class CudaBlockMat : public AbstractCsrMat
{
public:
    CudaBlockMat();

    /*!
     * \note  WARNING device deep copy is incomplete.
     * \brief returns a deep copy of the CudaMat
     * \return a deep copy of the CudaMat
     */
    CudaBlockMat<T> deepCopy();

    /*!
     * \brief Only DEEP copy's the CSR stuff. Leaves the data blocks alone
     * \return
     */
    CudaBlockMat<T> deepCopyCSR();

    void pushBackHost(CudaMat<T> val, size_t i, size_t j);
    /*!
     * \brief Set the dimensions of the composite blocks
     * \param n the width and height of the blocks
     */
    void setBlockDim(size_t n);
    void setBlockParam(denseParam_t param){*_blockParams=param;}
    denseParam_t getBlockParam(){return *_blockParams;}
    /*!
     * \brief getBlock from the value array index i
     * \param i value array index
     * \return
     */
    CudaMat<T> & getBlock(size_t i) const;
    /*!
     * \brief getBlock from a matrix position
     * \param i row
     * \param j column
     * \return the requested block if it exists. if it doesn't exist return CudaMat of size 0
     */
    virtual CudaMat<T> & getBlock(size_t row, size_t col, cusparseOperation_t op =  CUSPARSE_OPERATION_NON_TRANSPOSE);

    //CudaMat<T> & getBlockCSC(size_t row, size_t col){return getBlock(col,row,CUSPARSE_OPERATION_NON_TRANSPOSE);}

    /*!
     * \brief bring all blocks to device
     */
    void blocks2dev();

    /*!
     * \brief bring all blocks to host
     */
    void blocks2host();



    /*!
     * \brief gets the read only scalar value at the selected index.
     * \param i
     * \param j
     * \return
     */
    T getVal(size_t i,size_t j);

    /*!
     * \note IMPLEMENTATION INCOMPLETE
     * \brief This function returns the requested block if it exists.  If it does not exist the block is created and returned.
     * \param i row
     * \param j column
     * \return the requested block if it exists. if it doesn't exist return a new CudaMat of size specifed by setBlockDim()
     */
    CudaMat<T> & getCreateBlock(size_t i, size_t j);

    CudaBlockMat<T> transposeHost();

    void printHostValues();
    void printHostValuesMathematica();


    virtual void printHostData(int i);

    /*!
     * \brief getSparsityMatrix generates a CudaMat that shows the NNZ of each element in the matrix
     * \return
     */
    CudaMat<int> getSparsityMatrix();

    /*!
     * \brief gets the total number of values in a row
     * \return
     */
    int valuesPerRow(){ return rows()*_blockParams->rows;}

    /*!
     * \brief gets the total number of values in a column
     * \return
     */
    int valuesPerCol(){ return cols()*_blockParams->cols;}


protected:
    void resetData();
    void host2devData();
    void dev2hostData();

    /*!
     * \brief the host side representation of the value data. represented as a vector of CudaMat<T>
     */
    std::shared_ptr<std::vector<CudaMat<T > > > h_blockVal;
    /*!
     * \brief the device side representation of the value data.  represented as a thrust::device_vector of T array pointers
     */
    std::shared_ptr<thrust::device_vector<T*> > d_blockVal;

    std::shared_ptr<denseParam_t> _blockParams;

    CudaMat<T> _zero;

};

#endif // CUDABLOCKMAT_H
