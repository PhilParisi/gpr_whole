#ifndef CUDAMAT_H
#define CUDAMAT_H

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cublas_v2.h>
//#include <cusolverDn.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <memory>
#include <stdexcept>
#include "abstractmat.h"
#include <atomic>
#include <cusolverDn.h>
//#include <nvToolsExt.h>

#define ss_x 0
#define ss_y 1
#define ss_z 2
namespace ss {
namespace idx {
  enum column {x,y,z};
  __host__ __device__ size_t colMajor( size_t i, size_t j , size_t rows);  //!< \brief get the index of the ith row and jth colum of a col major matrix;
  __host__ __device__ size_t xVect( size_t i, size_t rows); //!< \brief get the index the i'th x element at the specified index;
  __host__ __device__ size_t yVect( size_t i, size_t rows); //!< \brief get the index the i'th y element at the specified index;
  __host__ __device__ size_t zVect( size_t i, size_t rows); //!< \brief get the index the i'th z element at the specified index;
}
}

/*!
 * \brief This enum specifies a memory object should be initialize on the host or device
 */
enum memType {host,dev};


/*!
 * \brief Represents a set of params that for a dense matrix
 */
struct denseParam_t
{

  /*!
   * \brief Basic constructor,  does nothing
   */
  denseParam_t(){return;}
  /*!
   * \brief denseParam_t constructor that specifies an inital matrix size
   * \param rows_in the number of rows your matrix should have
   * \param cols_in the number of columns your matrix should have
   */
  denseParam_t(size_t rows_in, size_t cols_in){
    rows = rows_in;
    cols = cols_in;
  }
  /*!
   * \brief Comparison operator
   * \param y
   * \return
   */
  bool operator ==(const denseParam_t &y) {
      return (rows==y.rows&&cols==y.cols);
  }
  /*!
   * \brief getRows returns the number of rows in a matrix.
   * This can be modified using a CUBLAS operator if you want to treat the matrix
   * as a transpose or a complex conjugate
   * \param op the cublasOperation_t of how you want to interpret the matrix
   * \return the number of rows in a matrix.
   */
  size_t getRows(cublasOperation_t op){
    switch (op) {
    case CUBLAS_OP_N:
      return rows;
    case CUBLAS_OP_T:
      return cols;
    case CUBLAS_OP_C:
      return cols;
    }
  }
  /*!
   * \brief getCols returns the number of colums in a matrix.
   * \param op the cublasOperation_t of how you want to interpret the matrix
   * \return number of colums in a matrix.
   */
  size_t getCols(cublasOperation_t op){
    switch (op) {
    case CUBLAS_OP_N:
      return cols;
    case CUBLAS_OP_T:
      return rows;
    case CUBLAS_OP_C:
      return rows;
    }
  }

  // member variables
  size_t rows; //!< \brief the number of rows of the raw dense matrix
  size_t cols; //!< \brief the number of columns of the raw dense matrix
};

/*!
 * \brief Contains the metadata corresponding to the cudamat  (see metatadata for usage)
 */
struct metadata_t
{
    int nnzCached;
    bool is_zero;
    metadata_t(){
      nnzCached = -1;  // initialize to -1 to indicate that cache has not been computed
      is_zero = false;
    }
};

/*!
 * \brief A host device representation of a cudamat's metadata
 */
class metadata{
  void host2dev();
  void dev2host();
private:
  metadata_t *host_data;
  metadata_t *device_data;
};

template <class T>
class CudaMat: public AbstractMat
{
public:
    CudaMat();
    /*!
     * \brief This constructor fully initializes the CudaMat on either the host or device
     * \param rows number of rows in the matrix
     * \param cols number of cols in the matrix
     * \param type the type of memory you want to initalize (see the memType enum)
     */
    CudaMat(size_t rows,size_t cols, memType type = host);

    CudaMat(denseParam_t params,memType type = host);

    CudaMat(const CudaMat& other);
    ~CudaMat();

    ///--------------
    /// memory management
    /// -------------

    /*!
     * \brief Set the matrix size to the specified rows and cols
     * \note does not initalize any data vectors. you must call initDev or initHost first
     * \param rows number of rows in the matrix
     * \param cols rumber of cols in the matrix
     */
    void reset( size_t rows,size_t cols);
    /*!
     * \brief initialize an empty matrix in device memory according to the set rows and cols
     */
    void initDev();
    /*!
     * \brief initialized an identity matrix on the device
     * \param ld the rows and columns of the identity
     */
    void initDevIdentity(size_t ld);
    /*!
     * \brief initialize an empty matrix in hoset memory according to the set rows and cols
     */
    void initHost();
    /*!
     * \brief syncronize the Device data to host memory
     */
    void dev2host();
    /*!
     * \brief syncronize the host memory to the device
     */
    void host2dev();

    /*!
     * \brief returns a deep copy of the CudaMat
     * this function allows you to do this type of operation safely
     * \code{.cpp}
     * CudaMat<float> x1;
     * CudaMat<float> x2(2,2);
     * // the value of x1 WILL be changed when x2 is changed
     * x1=x2;
     * x2(0,0)=88;
     *
     * // the value of x1 will NOT be changed when x2 is changed
     * x1=x2.deepcopy();
     * x2(0,0)=99;
     *
     * \endcode
     * \return a deep copy of the CudaMat
     */
    CudaMat<T> deepCopy();


    ///--------------
    /// dimensionality
    /// -------------

    /*!
     * \brief rows
     * \return
     */
    size_t rows(){ return _denseParam->rows; }
    size_t cols(){ return _denseParam->cols; }
    size_t ld(){return rows(); }
    size_t size(){return rows()*cols(); }

    void reserveDev(size_t n);

    void addRowsDev(size_t n);
    void addColsDev(size_t n);
    void insertDev(CudaMat<T> in, size_t i, size_t j);

    ///--------------
    /// accessors
    /// -------------

    /*!
     * \brief access the memory at the specified location in HOST memory
     * \param i the ROW index
     * \param j the Column index
     * \return a reference to the requested data on the HOST
     */
    T & val(size_t i,size_t j);
    /*!
     * \brief treat the cudamat as a 1d vector and access the memory at the specified location in HOST memory
     * \param i the i'th element of the data vector treated as a 1d vector.
     * \return a reference to the requested data on the HOST
     */
    T & val(size_t i);
    /*!
     * \brief redefine the () to wrap val(size_t i,size_t j)
     * \param i
     * \param j
     * \return a reference to the requested data on the HOST
     */
    T & operator ()(size_t i, size_t j);
    /*!
     * \brief treat the cudamat as a 1d vector and access the memory at the specified location in HOST memory
     * \param i the i'th element of the data vector treated as a 1d vector.
     * \return a reference to the requested data on the HOST
     */
    T & operator ()(size_t i);

    T & x(size_t i);
    T & y(size_t i);
    T & z(size_t i);
    /*!
     * \brief returns a raw pointer to an array of data in DEVICE memory for compatibilty with cuda kernals and cuda API
     * \return a pointer to a basic c array of the data in DEVICE memory
     */
    T * getRawDevPtr();

    ///--------------
    /// debugging
    /// -------------

    /*!
     * \brief print out data in host memory
     */
    void printHost();
    void printMathematica();
    void printHostRawData();
    /*!
     * \brief print out data in device memory
     * \note WARNING: thic can be expensive because it will need to syncronze host and dev memory
     */
    void printDev();
    ///--------------
    /// operators
    /// -------------


    /*!
     * \brief operator * multiplies two CudaMat togeter and returns the result
     * \param the matrix you want to mult by
     * \return the matrix multiplication of this * other
     */
    CudaMat<T> operator *(CudaMat<T> & other);
    /*!
     * \brief operator += adds a CudaMat to a given CudaMat
     * this ends up being a wrapper for add(*this,other)
     * \note this operation is less memory intensive than Cudamat + Cudamat and should be used whenever practical
     * \param other the value you want to add to your CudaMat
     * \return
     */
    CudaMat<T> & operator +=(CudaMat<T> & other);

    /*!
     * \brief operator -= subtracts a CudaMat from a given CudaMat
     * this ends up being a wrapper for add(*this,other,-1)
     * \note this operation is less memory intensive than Cudamat - Cudamat and should be used whenever practical
     * \param other the value you want to add to your CudaMat
     * \return
     */
    CudaMat<T> & operator -=(CudaMat<T> & other);

    /*!
     * \brief inverts the CudaMat object
     */
    void inv();

    /*!
     * \brief transpose transposes the matrix in place.
     * \note only vaid on square matracies!
     * \throws std::out_of_range if rows!=cols
     */
    void transpose();

    /*!
     * \brief returns the number of non zero elements of a matrix.
     * an element is considered to be zero below a
     * \param thresh elements below this number will be considered zero
     * \return the number of non zero elements
     */
    int nnz(float thresh);

    /*!
     * \brief if the number of nonzero elements is known by some external process.
     * This value for NNZ will be cached and can be accessed via nnzCached().
     * \param nnz
     */
    void setNNZ(int nnz){_metadata->nnzCached=nnz;}

    /*!
     * \brief Returns the last known number of non zero (NNZ) elements
     * \return
     */
    int nnzCached();

    /*!
     * \brief set this matrix a zero matrix
     */
    void setZero(){reset(0,0);_metadata->is_zero=true;}

    /*!
     * \brief check if this matrix is a zero matrix
     * \return a CudaMat of size (0,0) with the is_zero metata flag set to true
     */
    bool isZero(){return _metadata->is_zero;}

    /*!
     * \brief retuns a copy of the dense params.  Useful for initalizing another dense matrix
     * \return a copy of the dense params
     */
    denseParam_t getDenseParam(){return *_denseParam;}

    /*!
     * \brief this function allows you to link the parameters to that of other cudamats
     * this function is used by CudaBlockMat for example
     */
    void setDenseParam(std::shared_ptr<denseParam_t> in){_denseParam=in;}

    /*!
     * \brief get the cublas handle used by ALL cudamats
     * \return a reference to the cublas handle
     */
    cublasHandle_t & getHandle(){return _handle;}
    static cusolverDnHandle_t & getCusolverHandle(){
      if(_cusolverHandle==nullptr){
          cusolverDnCreate(&_cusolverHandle);
      }
      return _cusolverHandle;
    }

    /*!
     * \brief Returns the number of active cudamats created by sensorstream in your application.
     * this is usefull to make sure cudamats are being properly created or deleated.
     * \return the number of active cudamats created by sensorstream in your application.
     */
    static size_t getNumObjects(){return _numCudamats;}

    /*!
     * \brief check if there is enough room on the device to allocate a new cudamat
     * \return true if there is space available
     */
    bool canAlocDev();


protected:
    std::shared_ptr<denseParam_t> _denseParam;
    std::shared_ptr<thrust::device_vector<T> > _dDat;
    std::shared_ptr<thrust::host_vector<T>   > _hDat;
    std::shared_ptr<metadata_t> _metadata;

    static cublasHandle_t _handle;
    static cusolverDnHandle_t _cusolverHandle;
    static std::atomic_ulong _numCudamats;

};

//
// cublas core operations
//


/*!
 * \brief mult returns alpha*(a*b)
 * \param a input matrix
 * \param b input matrix
 * \param opA should A be transposed?
 * \param opB should B be transposed?
 * \param alpha constant multiple of the result a*b
 * \param beta not currently implemented
 * \return the value alpha*(a*b)
 */
CudaMat<float> mult(CudaMat<float> a, CudaMat<float> b ,
             cublasOperation_t opA = CUBLAS_OP_N, cublasOperation_t opB = CUBLAS_OP_N,
             float alpha = 1.0f, float beta = 0.0f);

/*!
 * \brief multiply each element in a by alpha
 * \param a [input/output]
 * \param alpha
 */
void mult(CudaMat<float> a, float alpha);

/*!
 * \brief computes: y=y+alpha*x
 * \param y input/output: the cudaMat you want to add (alpha * x) to
 * \param x input only:  the cudaMat to add to y
 * \param alpha a constant scalar to multiply to x
 * \return
 */
template <class T>
void add(CudaMat<T> & y, CudaMat<T> x, float alpha=1, int incy = 1, int incx = 1);

/*!
 * \brief add a constant to a cudamat
 * \param x input matrix
 * \param alpha a constant to add to x
 */
void add(CudaMat<float> x, float alpha);



/*!
 * \brief allows you to add two cudamats and specify cublasOperation_t to specify if
 * you want to treat the matracies as tranpose or complex conjugate
 * \param y the frist matrix
 * \param x the second matrix
 * \param alpha
 * \param opY
 * \param opX
 */
void addTrans(CudaMat<float> y, CudaMat<float> x, float alpha=1, cublasOperation_t opY = CUBLAS_OP_N, cublasOperation_t opX = CUBLAS_OP_N);

void invert(CudaMat<float> & in);

//
// Templates for calling cuda and cublas
//

#define cudacall(call)                                                                                                          \
    do                                                                                                                          \
    {                                                                                                                           \
        cudaError_t err = (call);                                                                                               \
        if(cudaSuccess != err)                                                                                                  \
        {                                                                                                                       \
            throw std::runtime_error("cuda error");                                                                             \
            fprintf(stderr,"CUDA Error:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, cudaGetErrorString(err));    \
            cudaDeviceSynchronize();                                                                                            \
            cudaDeviceReset();                                                                                                  \
        }                                                                                                                       \
    }                                                                                                                           \
    while (0)

#define cublascall(call)                                                                                        \
    do                                                                                                          \
    {                                                                                                           \
        cublasStatus_t status = (call);                                                                         \
        if(CUBLAS_STATUS_SUCCESS != status)                                                                     \
        {                                                                                                       \
            fprintf(stderr,"CUBLAS Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status);     \
            cudaDeviceSynchronize();                                                                            \
            cudaDeviceReset();                                                                                  \
            throw std::runtime_error("cuda error");                                                             \
        }                                                                                                       \
                                                                                                                \
    }                                                                                                           \
    while(0)

#define cusolvercall(call)                                                                                      \
    do                                                                                                          \
    {                                                                                                           \
        cusolverStatus_t status = (call);                                                                       \
        if(CUSOLVER_STATUS_SUCCESS != status)                                                                   \
        {                                                                                                       \
            fprintf(stderr,"CUSOLVER Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status);   \
            cudaDeviceSynchronize();                                                                            \
            cudaDeviceReset();                                                                                  \
            throw std::runtime_error("cuda error");                                                             \
        }                                                                                                       \
                                                                                                                \
    }                                                                                                           \
    while(0)




#endif // CUDAMAT_H
