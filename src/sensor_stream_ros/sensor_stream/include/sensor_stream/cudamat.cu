#include "cudamat.h"

// initialize static members
template <class T>
cublasHandle_t CudaMat<T>::_handle = nullptr;

template <class T>
std::atomic_ulong CudaMat<T>::_numCudamats(0);

template <class T>
cusolverDnHandle_t CudaMat<T>::_cusolverHandle = nullptr;


//idx defs
__host__ __device__ size_t ss::idx::colMajor(size_t i, size_t j , size_t rows){
  return j*rows+i;
}

__host__ __device__ size_t ss::idx::xVect(size_t i, size_t rows){
  return ss::idx::colMajor(i,0,rows);
}

__host__ __device__ size_t ss::idx::yVect(size_t i, size_t rows){
  return ss::idx::colMajor(i,1,rows);
}

__host__ __device__ size_t ss::idx::zVect(size_t i, size_t rows){
  return ss::idx::colMajor(i,2,rows);
}

template <class T>
bool CudaMat<T>::canAlocDev(){
  ///cudaDeviceSynchronize();
  size_t free, total;
  cudaMemGetInfo( &free, &total );
  size_t required = getDenseParam().rows*getDenseParam().cols*sizeof (T);
  return required + 3e+8 < free;
}


//cudamat function defs
template <class T>
CudaMat<T>::CudaMat()
{
    _denseParam.reset(new denseParam_t);
    _metadata.reset(new metadata_t);
    if(_handle==nullptr){
        cublascall(cublasCreate_v2(&_handle));
        cublascall(cublasSetMathMode(_handle, CUBLAS_TENSOR_OP_MATH));  // activate tensor cores if available
    }
    _numCudamats++;
    return;
}

template <class T>
CudaMat<T>::CudaMat(size_t rows,size_t cols, memType type)
{
    reset(rows,cols);
    if(type==host)
        initHost();
    if(type==dev)
        initDev();

    if(_handle==nullptr){
        cublascall(cublasCreate_v2(&_handle));
        cublascall(cublasSetMathMode(_handle, CUBLAS_TENSOR_OP_MATH));  // activate tensor cores if available
    }
    _numCudamats++;
}

template <class T>
CudaMat<T>::CudaMat(denseParam_t params, memType type){

  reset(params.rows,params.cols);
  if(type==host)
      initHost();
  if(type==dev)
      initDev();

  if(_handle==nullptr){
      cublascall(cublasCreate_v2(&_handle));
      cublascall(cublasSetMathMode(_handle, CUBLAS_TENSOR_OP_MATH));  // activate tensor cores if available
  }
  _numCudamats++;
}

template <class T>
CudaMat<T>::CudaMat(const CudaMat& other){
   _numCudamats++;
   _denseParam=other._denseParam;
   _metadata = other._metadata;
   _dDat=other._dDat;
   _hDat=other._hDat;
}


template <class T>
CudaMat<T>::~CudaMat(){
     _numCudamats--;
}

template <class T>
void CudaMat<T>::reset(size_t rows,size_t cols){
    _dDat.reset(new thrust::device_vector<T>);
    _hDat.reset(new thrust::host_vector<T>  );
    _denseParam.reset(new denseParam_t);
    _metadata.reset(new metadata_t);
//    _rows.reset(new size_t);
//    _cols.reset(new size_t);
    _denseParam->rows=rows;
    _denseParam->cols=cols;
}

template <class T>
void CudaMat<T>::initDev(){
  if(canAlocDev()){
    _dDat->resize(this->size());
  }else {
    throw std::runtime_error("not enough GPU memory to complete initDev()");
  }

}


__global__ void identityKernel(float * x, size_t ld){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int outIdx = j*ld+i;
    if(i<ld && j<ld){
        if(i==j){
            x[outIdx]=1;
        }
        else {
            x[outIdx]=0;
        }
    }
}

template <>
void CudaMat<float>::initDevIdentity(size_t ld){
    this->reset(ld,ld);
    this->initDev();
    dim3 threads_per_block (32, 32, 1);
    dim3 number_of_blocks ((this->rows() / threads_per_block.x) + 1, (this->cols() / threads_per_block.y) + 1, 1);

    identityKernel <<< number_of_blocks, threads_per_block >>> (this->getRawDevPtr(),this->ld());
}

template <class T>
void CudaMat<T>::initHost(){
    _hDat->resize(this->size());
}

template <class T>
void CudaMat<T>::dev2host(){    //size_t offset = posJ * ldx + posI;

    if(this->size()==_dDat->size())
        *_hDat=*_dDat;
    else{
        std::cout<< _dDat->size() <<std::endl;
        throw std::out_of_range("Size missmatch! device data size != CudaMat size.  Has device data been initialized or syncronized yet?");
    }

}

template <class T>
void CudaMat<T>::host2dev(){
    if(this->size()==_hDat->size()){
      if(canAlocDev()){
        *_dDat=*_hDat;
        cudacall(cudaGetLastError());
      }else {
        throw std::runtime_error("not enough GPU memory to complete host2dev()");
      }
    }
    else{
        throw std::out_of_range("Size missmatch! host data size != CudaMat size.  has host data been initialized or syncronized yet?");
    }
}


template <class T>
CudaMat<T> CudaMat<T>::deepCopy(){
    CudaMat<T> copy;
    copy.reset(this->rows(),this->cols());
    *copy._dDat=*_dDat;
    *copy._hDat=*_hDat;
    *copy._denseParam=*_denseParam;
    *copy._metadata = *_metadata;
    return copy;
}

template <class T>
void CudaMat<T>::reserveDev(size_t n){
    _dDat->reserve(n);
}

template <class T>
void CudaMat<T>::addRowsDev(size_t n){
    for(size_t i = cols(); i>0 ; i--){
        unsigned int index =ld()*(i);
        _dDat->insert(_dDat->begin()+index,n,0);
    }
    _denseParam->rows=_denseParam->rows+n;
    return;
}

template <class T>
void CudaMat<T>::addColsDev(size_t n){
    _denseParam->cols=_denseParam->cols+n;
    _dDat->resize(this->size());
    return;
}

__global__ void insertKernel(float * x, size_t ldx, float * in, size_t ldin, size_t colsin, size_t posI, size_t posJ){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    size_t xIndex = (posJ+j)*ldx+posI+i;
    size_t inIndex = j*ldin+i;
    if(i<ldin && j<colsin){
        x[xIndex]=in[inIndex];
    }
}
template <>
void CudaMat<float>::insertDev(CudaMat<float> in, size_t i, size_t j){
    dim3 threads_per_block (32, 32, 1);
    dim3 number_of_blocks ((in.rows() / threads_per_block.x) + 1, (in.cols() / threads_per_block.y) + 1, 1);

    insertKernel <<< number_of_blocks, threads_per_block >>> (this->getRawDevPtr(),this->ld(),in.getRawDevPtr(),in.ld(),in.cols(),i,j);
}

template <class T>
T & CudaMat<T>::val(size_t i, size_t j){
  if(i < _denseParam->rows && j < _denseParam->cols){
    if(this->size()==_hDat->size())
        return _hDat->operator[]( ss::idx::colMajor(i,j,rows()) );
    else{
        throw std::out_of_range("Size missmatch! host data size != CudaMat size.  has host data been initialized or syncronized yet?");
    }
  }
  else{
    throw std::out_of_range("attempted to acccess an out of bounds value");
  }
}

template <class T>
T & CudaMat<T>::val(size_t i){
  if(i < _hDat->size()){
    if(this->size()==_hDat->size())
        return _hDat->operator[](i);
    else{
        throw std::out_of_range("Size missmatch! host data size != CudaMat size.  has host data been initialized or syncronized yet?");
    }
  }
  else{
    throw std::out_of_range("attempted to acccess an out of bounds value");
  }
}


template <class T>
T & CudaMat<T>::operator ()(size_t i, size_t j){

    return val(i,j);
}

template <class T>
T & CudaMat<T>::operator ()(size_t i){

    return val(i);
}

template  <class T>
T & CudaMat<T>::x(size_t i){
  if(cols()<=0){
    std::out_of_range("attempted to access x vector a cudamat with 0 rows");
  }
  return val( ss::idx::xVect(i,rows()) );
}

template  <class T>
T & CudaMat<T>::y(size_t i){
  if(cols()<2){
    std::out_of_range("attempted to access y vector a cudamat with < 2 rows");
  }
  return val( ss::idx::yVect(i,rows()) );
}

template  <class T>
T & CudaMat<T>::z(size_t i){
  if(cols()<3){
    std::out_of_range("attempted to access z vector a cudamat with < 3 rows");
  }
  return val( ss::idx::zVect(i,rows()) );
}

template <class T>
T * CudaMat<T>::getRawDevPtr(){
    if(this->size()!=_dDat->size())
        throw std::out_of_range("Size missmatch! device data size != CudaMat size.  Has device data been initialized or syncronized yet?");
    return thrust::raw_pointer_cast(_dDat->data());
}

template <>
void CudaMat<float>::printHost(){
    if(this->size()==0){
        printf("empty matrix\n");
    }
    for(size_t i = 0; i < rows() ; i++){
        for(size_t j=0; j < cols() ; j++){
             printf ("%10.4f ", val(i,j));
        }
        std::cout<< std::endl;
    }
    std::cout<< std::endl;
}

template <>
void CudaMat<int>::printHost(){
    if(this->size()==0){
        printf("empty matrix\n");
    }
    for(size_t i = 0; i < rows() ; i++){
        for(size_t j=0; j < cols() ; j++){
          printf("|%8d|", val(i,j));
        }
        std::cout<< std::endl;
    }
    std::cout<< std::endl;
}


template <class T>
void CudaMat<T>::printMathematica(){
    if(this->size()==0){
        printf("empty matrix\n");
    }
    printf ("{\n");
    for(size_t i = 0; i < rows() ; i++){
        printf ("   {");
        for(size_t j=0; j < cols() ; j++){
             printf ("%15.10f", val(i,j));
             if(j<cols()-1){
                 printf (",");
             }
        }
        printf ("}");
        if(i<cols()-1){
            printf (",");
        }
        std::cout<< std::endl;
    }
    printf ("}");
    std::cout<< std::endl;
}

template <class T>
void CudaMat<T>::printHostRawData(){
    for(size_t i = 0; i < this->size() ; i++){
        printf ("%3f ", _hDat->operator[](i));
    }
    std::cout<< std::endl<< std::endl;
}

template <>
CudaMat<float> CudaMat<float>::operator *(CudaMat<float> & other){
    return mult(*this,other);
}

template <class T>
CudaMat<T> & CudaMat<T>::operator +=(CudaMat<T> &other){
    add(*this,other);
    return *this;
}

template <class T>
CudaMat<T> & CudaMat<T>::operator -=(CudaMat<T> &other){
    add(*this,other,-1);
    return *this;
}

template <>
void CudaMat<float>::inv(){
    invert(*this);
}

__global__ void transposeKernel(float * x, size_t ldx){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    size_t index = j*ldx+i;
    size_t swapIdx = i*ldx+j;
    if(i<ldx && j<ldx && i>j){
        float temp = x[swapIdx];
        x[swapIdx]=x[index];
        x[index]=temp;
    }
}

template <>
void CudaMat<float>::transpose(){
    if(rows()==cols()){
        dim3 threads_per_block (32, 32, 1);
        dim3 number_of_blocks ((rows() / threads_per_block.x) + 1, (cols() / threads_per_block.y) + 1, 1);

        transposeKernel <<< number_of_blocks, threads_per_block >>> (this->getRawDevPtr(),this->ld());

    }else{
        throw std::out_of_range("CudaMat<T>::transpose() is only valid for square matracies");
    }
}


__global__ void zeroKernel(float * x, size_t ldx, size_t cols , float thresh, int * nonzero){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    size_t index = j*ldx+i;
    if(i<ldx && j<cols){
        if(fabs(x[index]) > thresh){
            atomicAdd(nonzero,1);
        }
    }
}

template <>
int CudaMat<float>::nnz(float thresh){
    int *nnz;
      cudaMallocManaged(&nnz, 4);
      *nnz = 0;
    dim3 threads_per_block (32, 32, 1);
    dim3 number_of_blocks ((rows() / threads_per_block.x) + 1, (cols() / threads_per_block.y) + 1, 1);
    zeroKernel <<< number_of_blocks, threads_per_block >>> (this->getRawDevPtr(),
                                                            this->ld(),this->cols(),
                                                            thresh,nnz);
    cudaStreamSynchronize(0);
    int out = *nnz;
    cudaFree(nnz);
    _metadata->nnzCached = out;
    return out;


}

template <>
int CudaMat<float>::nnzCached(){
  //return nnz(0.0001f);
  //cudaStreamSynchronize(0);
  if(_metadata->nnzCached == -1){
    return size();
  }
  return _metadata->nnzCached;
}

//
//  BLAS operations
//

CudaMat<float> mult(CudaMat<float> a, CudaMat<float> b ,
             cublasOperation_t opA, cublasOperation_t opB,
             float alpha, float beta){

  if(a.isZero()||b.isZero()){
    CudaMat<float> c;
    c.setZero();
    return c;
  }

    int m = 0;
    if(opA==CUBLAS_OP_N)
        m=a.rows();
    else
        m=a.cols();

    int n = 0;
    if(opB==CUBLAS_OP_N)
        n=b.cols();
    else
        n=b.rows();

    int k = 0;
    if(opA==CUBLAS_OP_N)
        k=a.cols();
    else
        k=a.rows();


    CudaMat<float> c(m,n,dev);
    cublascall(cublasSgemm(a.getHandle(),opA,opB,
                           m, n, k,
                           &alpha, a.getRawDevPtr(), a.ld(),
                                   b.getRawDevPtr(), b.ld(),
                           &beta,  c.getRawDevPtr(), c.ld()));

    return c;
}



__global__ void multKernel(float * a, size_t rows, size_t cols, float alpha){
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t j = blockIdx.y * blockDim.y + threadIdx.y;
  size_t index = j*rows+i;//ss::idx::colMajor(i,j,rows);
  if(i<rows && j<cols){
    a[index]=a[index]*alpha;
  }
}

void mult(CudaMat<float> a, float alpha){
  dim3 threads_per_block (32, 32, 1);
  dim3 number_of_blocks ((a.rows() / threads_per_block.x) + 1, (a.cols() / threads_per_block.y) + 1, 1);
  multKernel <<< number_of_blocks, threads_per_block >>>(a.getRawDevPtr(),a.rows(),a.cols(),alpha);
}



template <>
void add(CudaMat<float> & y, CudaMat<float> x, float alpha, int incy, int incx){
  if(x.isZero()){
    return;
  }
  else if(y.isZero()){
    y=x.deepCopy();
    mult(y,alpha);
    return;
  }else{
    assert(y.rows() == x.rows() && y.cols()== x.cols());
    cublascall(cublasSaxpy(y.getHandle(),y.size(),&alpha,x.getRawDevPtr(),incx,y.getRawDevPtr(),incy));
    return;
  }

}

__global__ void addScalarKernel(float * x, size_t ld, size_t cols, float alpha){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    size_t xIndex;
    xIndex = j*ld+i;

    if(i<ld && j<cols){
        x[xIndex]=x[xIndex]+alpha;
    }
}


void add(CudaMat<float> x, float alpha){
  dim3 threads_per_block (32, 32, 1);
  dim3 number_of_blocks ((x.rows() / threads_per_block.x) + 1, (x.cols() / threads_per_block.y) + 1, 1);

  addScalarKernel <<< number_of_blocks, threads_per_block >>> (x.getRawDevPtr(),x.ld(),x.cols(),alpha);
}




template <>
void add(CudaMat<int> & y, CudaMat<int> x, float alpha, int incy, int incx){
  throw std::invalid_argument("add is not yet defined for CudaMat<int>");
}

__global__ void addKernel(float * y, float * x, size_t ld, size_t cols, float alpha, cublasOperation_t opY, cublasOperation_t opX){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    size_t yIndex;
    if(opY==CUBLAS_OP_N)
        yIndex = j*ld+i;
    else
        yIndex = i*ld+j;


    size_t xIndex;
    if(opX==CUBLAS_OP_N)
        xIndex = j*ld+i;
    else
        xIndex = i*ld+j;


    if(i<ld && j<cols){
        y[yIndex]=x[xIndex]*alpha+y[yIndex];
    }
}

void addTrans(CudaMat<float> y, CudaMat<float> x, float alpha, cublasOperation_t opY, cublasOperation_t opX){
  if(x.isZero()){
    return;
  }
  else if(y.isZero()){
    y=x.deepCopy();
    mult(y,alpha);
    return;
  }else{
    dim3 threads_per_block (32, 32, 1);
    dim3 number_of_blocks ((y.rows() / threads_per_block.x) + 1, (y.cols() / threads_per_block.y) + 1, 1);

    addKernel <<< number_of_blocks, threads_per_block >>> (y.getRawDevPtr(),x.getRawDevPtr(),y.ld(),y.cols(),alpha,opY,opX);
  }
}


//template <class T>
void invert(CudaMat<float> & in){
    CudaMat<float> out(in.rows(),in.cols(),dev);
    int batchSize=1;

    int *P, *INFO;

    int lda = in.ld();
    int n = in.rows();

    cudacall(cudaMalloc(&P, n * batchSize * sizeof(int)));
    cudacall(cudaMalloc(&INFO,  batchSize * sizeof(int)));


    float **A=(float **)malloc(batchSize*sizeof(float *));
    float **A_d;
    cudacall(cudaMalloc(&A_d,batchSize*sizeof(float *)));
    A[0]=in.getRawDevPtr();

    cudacall(cudaMemcpy(A_d,A,batchSize*sizeof(float *),cudaMemcpyHostToDevice));

    cublascall(cublasSgetrfBatched(in.getHandle(),n,A_d,lda,P,INFO,batchSize));


    int INFOh[batchSize];
    cudacall(cudaMemcpy(INFOh,INFO,batchSize*sizeof(int),cudaMemcpyDeviceToHost));

    for (int i = 0; i < batchSize; i++)
      if(INFOh[i]  != 0)
      {
        fprintf(stderr, "Factorization of matrix %d Failed: Matrix may be singular\n", i);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
      }


    float **C = (float **)malloc(batchSize*sizeof(float *));
    float **C_d;
    cudacall(cudaMalloc(&C_d,batchSize*sizeof(float *)));

    C[0] = out.getRawDevPtr();
    for (int i = 1; i < batchSize; i++)
      C[i] = C[i-1] + (n*n);

    cudacall(cudaMemcpy(C_d,C,batchSize*sizeof(float *),cudaMemcpyHostToDevice));

    cublascall(cublasSgetriBatched(in.getHandle(),n,(const float **)A_d,lda,P,C_d,lda,INFO,batchSize));

    cudacall(cudaMemcpy(INFOh,INFO,batchSize*sizeof(int),cudaMemcpyDeviceToHost));

    for (int i = 0; i < batchSize; i++)
      if(INFOh[i] != 0)
      {
        fprintf(stderr, "Inversion of matrix %d Failed: Matrix may be singular\n", i);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
      }

    cudaFree(A_d); free(A);
    cudaFree(C_d); free(C);
    cudaFree(P); cudaFree(INFO);

    in=out;

}





//
// declare templates
//

template class CudaMat<float>;
template class CudaMat<int>;









// just a little main for testing
//int main(){
//    CudaMat<float> x1(2,2);
//    CudaMat<float> x2(2,2);
//    CudaMat<float> x3(2,2);
//    CudaMat<float> result;

//    x1(0,0)=-1;
//    x1(1,0)=3.0/2.0;
//    x1(0,1)=1;
//    x1(1,1)=-1;
//    x1.printHost();

//    x2(0,0)=5;
//    x2(1,0)=6;
//    x2(0,1)=7;
//    x2(1,1)=8;
//    x2.printHost();

//    x3(0,0)=9;
//    x3(1,0)=10;
//    x3(0,1)=11;
//    x3(1,1)=12;
//    x3.printHost();

//    x1.host2dev();
//    invert(x1);
//    x1.dev2host();
//    x1.printHost();

////    x1=x2.deepCopy();
////    x2(0,0)=99;
////    x1.printHost();
////    x2.printHost();


////    x1.host2dev();
////    x2.host2dev();
////    x3.host2dev();

////    result = x1*x2*x3;//mult(x1,x2);
////    result.dev2host();
////    result.printHost();

////    x1=x2.deepCopy();
////    x2(0,0)=99;

////    //add(x1,x2);
////    x1-=x2;
////    x1.dev2host();
////    x1.printHost();
//    return 0;
//}
