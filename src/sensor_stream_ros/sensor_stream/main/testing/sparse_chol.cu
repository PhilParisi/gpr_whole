#include <memory>  // std::shared_ptr
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <cusparse_v2.h>
#include "../../include/sensor_stream/cudamat.h"
#include "../../include/sensor_stream/cudasparsemat.h"


//template <class T>
//class CudaSparseMat{
//public:
//    CudaSparseMat();
//    void reset(size_t dataSize=0);
//    void initDev();
//    void initHost();
//    void dev2host();
//    void host2dev();


//    void pushBackHost(T val, unsigned int i, unsigned int j);

//    void printHostRawData();
//    void printHostCsrRow();
//    void printHostCsrCol();
//    T val(size_t i, size_t j);

//    T * rawValPtr();
//    int * rawCsrRowPtr();
//    int * rawCsrColPtr();
//    size_t nnz();

//protected:
//    std::shared_ptr<thrust::device_vector<int> > d_csrRowPtr;
//    std::shared_ptr<thrust::device_vector<int> > d_csrColInd;
//    std::shared_ptr<thrust::device_vector<T> > d_csrVal;

//    std::shared_ptr<thrust::host_vector<int> > h_csrRowPtr;
//    std::shared_ptr<thrust::host_vector<int> > h_csrColInd;
//    std::shared_ptr<thrust::host_vector<T> > h_csrVal;

//    size_t _dataSize;
//    size_t _nnz; // number non zero
//};

//template <class T>
//CudaSparseMat<T>::CudaSparseMat()
//{
//    _nnz=0;
//}

//template <class T>
//void CudaSparseMat<T>::reset(size_t dataSize){
//    d_csrRowPtr.reset(new thrust::device_vector<int>);
//    d_csrColInd.reset(new thrust::device_vector<int>);
//    d_csrVal.reset(new thrust::device_vector<T>);

//    h_csrRowPtr.reset(new thrust::host_vector<int> );
//    h_csrColInd.reset(new thrust::host_vector<int> );
//    h_csrVal.reset(new thrust::host_vector<T> );

//    size_t _rows;
//    size_t _cols;

//    _dataSize=dataSize;
//}

//template <class T>
//void CudaSparseMat<T>::initDev(){
//    d_csrRowPtr->reserve(_dataSize);
//    d_csrColInd->reserve(_dataSize);
//    d_csrVal->reserve(_dataSize);
//}

//template <class T>
//void CudaSparseMat<T>::initHost(){
//    h_csrRowPtr->reserve(_dataSize);
//    h_csrColInd->reserve(_dataSize);
//    h_csrVal->reserve(_dataSize);
//}

//template <class T>
//void CudaSparseMat<T>::host2dev(){
//    *d_csrRowPtr=*h_csrRowPtr;
//    *d_csrColInd=*h_csrColInd;
//    *d_csrVal=*h_csrVal;
//}

//template <class T>
//void CudaSparseMat<T>::dev2host(){
//    *h_csrRowPtr=*d_csrRowPtr;
//    *h_csrColInd=*d_csrColInd;
//    *h_csrVal=*d_csrVal;
//}

//template <class T>
//void CudaSparseMat<T>::pushBackHost(T val, unsigned int i , unsigned int j){
//    _nnz++;
//    h_csrVal->push_back(val);
//    h_csrColInd->push_back(j);
//    if(i>h_csrRowPtr->size()-2||h_csrRowPtr->size()==0){
//        h_csrRowPtr->resize(i+2);
//        h_csrRowPtr->operator [](i)=h_csrVal->size()-1;  // index of last element in the val array
//    }
//    h_csrRowPtr->operator [](i+1)=_nnz+h_csrRowPtr->operator [](0);
//}

//template <class T>
//void CudaSparseMat<T>::printHostRawData(){
//    for(size_t i = 0; i < h_csrVal->size() ; i++){
//        printf ("%3f ", h_csrVal->operator[](i));
//    }
//    std::cout<< std::endl<< std::endl;
//}

//template <class T>
//void CudaSparseMat<T>::printHostCsrRow(){
//    for(size_t i = 0; i < h_csrRowPtr->size() ; i++){
//        printf ("%3i ", h_csrRowPtr->operator[](i));
//    }
//    std::cout<< std::endl<< std::endl;
//}

//template <class T>
//void CudaSparseMat<T>::printHostCsrCol(){
//    for(size_t i = 0; i < h_csrColInd->size() ; i++){
//        printf ("%3i ", h_csrColInd->operator[](i));
//    }
//    std::cout<< std::endl<< std::endl;
//}


//template <class T>
//T CudaSparseMat<T>::val(size_t i, size_t j){
//    size_t rowBegin = h_csrRowPtr->operator [](i);
//    size_t rowEnd;
//    if(i+1 < h_csrRowPtr->size()){
//        rowEnd=h_csrRowPtr->operator [](i+1);
//    }else{
//        rowEnd=h_csrVal->size()-1;
//    }

//    size_t valIdx;
//    for(valIdx=rowBegin; valIdx==j ; valIdx++){
//        if(valIdx>=rowEnd){
//            return T();  // value not found must be a zero
//        }
//    }
//    return h_csrVal->operator [](valIdx);
////    size_t rowIdx=h_csrRowPtr->operator [](i);
////    size_t valIdx=rowIdx;
////    for(valIdx=rowIdx; valIdx==j ; valIdx++){
////        if(valIdx>=h_csrVal->size()){
////            return T();  // value not found must be a zero
////        }
////    }
////    return h_csrVal->operator [](valIdcusparseMatDescr_tx);
//}

//template <class T>
//T * CudaSparseMat<T>::rawValPtr(){
//    return thrust::raw_pointer_cast(d_csrVal->data());
//}

//template <class T>
//int * CudaSparseMat<T>::rawCsrRowPtr(){
//    return thrust::raw_pointer_cast(d_csrRowPtr->data());
//}

//template <class T>
//int * CudaSparseMat<T>::rawCsrColPtr(){
//    return thrust::raw_pointer_cast(d_csrColInd->data());
//}

//template <class T>
//size_t CudaSparseMat<T>::nnz(){
//    return d_csrVal->size();
//}



int main(int argc, char *argv[])
{

    CudaSparseMat<float> testMat;
    testMat.reset(10);
    testMat.initHost();
    testMat.pushBackHost(4 ,0,0);
    testMat.pushBackHost(12 ,0,1);
    testMat.pushBackHost(-16 ,0,2);
    testMat.pushBackHost(12 ,1,0);
    testMat.pushBackHost(37 ,1,1);
    testMat.pushBackHost(-43 ,1,2);
    testMat.pushBackHost(-16 ,2,0);
    testMat.pushBackHost(-43 ,2,1);
    testMat.pushBackHost(98 ,2,2);

    testMat.printHostRawData();
    testMat.printHostCsrRow();
    testMat.printHostCsrCol();

    testMat.host2dev();

    CudaMat<float> x(3,1);
    x(0,0)=1;
    x(1,0)=1;
    x(2,0)=1;
    x.host2dev();

    CudaMat<float> y(3,1,dev);
    CudaMat<float> z(3,1,dev);

    float * d_x = x.getRawDevPtr();
    float * d_y = y.getRawDevPtr();
    float * d_z = z.getRawDevPtr();

    //Foo** f;
//    const int * d_csrRowPtr = const_cast<const int *>(testMat.rawCsrRowPtr());
//    const int * d_csrColInd = const_cast<const int *>(testMat.rawCsrColPtr());
    int * d_csrRowPtr = testMat.rawCsrRowPtr();
    int * d_csrColInd = testMat.rawCsrColPtr();
    float * d_csrVal = testMat.rawValPtr();
    size_t nnz= testMat.nnz();
    cusparseHandle_t handle = NULL;
    cusparseCreate(&handle);

    int m = 3;

    // Suppose that A is m x m sparse matrix represented by CSR format,
    // Assumption:
    // - handle is already created by cusparseCreate(),
    // - (d_csrRowPtr, d_csrColInd, d_csrVal) is CSR of A on device memory,
    // - d_x is right hand side vector on device memory,
    // - d_y is solution vector on device memory.
    // - d_z is intermediate result on device memory.

    cusparseMatDescr_t descr_M = 0;
    cusparseMatDescr_t descr_L = 0;
    csric02Info_t info_M  = 0;
    csrsv2Info_t  info_L  = 0;
    csrsv2Info_t  info_Lt = 0;
    int pBufferSize_M;
    int pBufferSize_L;
    int pBufferSize_Lt;
    int pBufferSize;
    void *pBuffer = 0;
    int structural_zero;
    int numerical_zero;
    const float alpha = 1.;
    const cusparseSolvePolicy_t policy_M  = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    const cusparseSolvePolicy_t policy_L  = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    const cusparseSolvePolicy_t policy_Lt = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L  = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseOperation_t trans_Lt = CUSPARSE_OPERATION_TRANSPOSE;

    // step 1: create a descriptor which contains
    // - matrix M is base-1
    // - matrix L is base-1
    // - matrix L is lower triangular
    // - matrix L has non-unit diagonal
    cusparseCreateMatDescr(&descr_M);
    cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);

    cusparseCreateMatDescr(&descr_L);
    cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_NON_UNIT);

    // step 2: create a empty info structure
    // we need one info for csric02 and two info's for csrsv2
    cusparseCreateCsric02Info(&info_M);
    cusparseCreateCsrsv2Info(&info_L);
    cusparseCreateCsrsv2Info(&info_Lt);

    // step 3: query how much memory used in csric02 and csrsv2, and allocate the buffer
    cusparseScsric02_bufferSize(handle, m, nnz,
        descr_M, d_csrVal, d_csrRowPtr, d_csrColInd, info_M, &pBufferSize_M);
    cusparseScsrsv2_bufferSize(handle, trans_L, m, nnz,
        descr_L, d_csrVal, d_csrRowPtr, d_csrColInd, info_L, &pBufferSize_L);
    cusparseScsrsv2_bufferSize(handle, trans_Lt, m, nnz,
        descr_L, d_csrVal, d_csrRowPtr, d_csrColInd, info_Lt,&pBufferSize_Lt);

    pBufferSize = max(pBufferSize_M, max(pBufferSize_L, pBufferSize_Lt));

    // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
    cudaMalloc((void**)&pBuffer, pBufferSize);

    // step 4: perform analysis of incomplete Cholesky on M
    //         perform analysis of triangular solve on L
    //         perform analysis of triangular solve on L'
    // The lower triangular part of M has the same sparsity pattern as L, so
    // we can do analysis of csric02 and csrsv2 simultaneously.

    cusparseScsric02_analysis(handle, m, nnz, descr_M,
        d_csrVal, d_csrRowPtr, d_csrColInd, info_M,
        policy_M, pBuffer);
    cusparseStatus_t status = cusparseXcsric02_zeroPivot(handle, info_M, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status){
       printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
    }

    cusparseScsrsv2_analysis(handle, trans_L, m, nnz, descr_L,
        d_csrVal, d_csrRowPtr, d_csrColInd,
        info_L, policy_L, pBuffer);

    cusparseScsrsv2_analysis(handle, trans_Lt, m, nnz, descr_L,
        d_csrVal, d_csrRowPtr, d_csrColInd,
        info_Lt, policy_Lt, pBuffer);

    // step 5: M = L * L'
    cusparseScsric02(handle, m, nnz, descr_M,
        d_csrVal, d_csrRowPtr, d_csrColInd, info_M, policy_M, pBuffer);
    status = cusparseXcsric02_zeroPivot(handle, info_M, &numerical_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status){
       printf("L(%d,%d) is zero\n", numerical_zero, numerical_zero);
    }
    cudacall(cudaDeviceSynchronize() );
    testMat.dev2host();
    testMat.printHostRawData();
    testMat.printHostCsrRow();
    testMat.printHostCsrCol();

    // step 6: solve L*z = x
    cusparseScsrsv2_solve(handle, trans_L, m, nnz, &alpha, descr_L,
       d_csrVal, d_csrRowPtr, d_csrColInd, info_L,
       d_x, d_z, policy_L, pBuffer);

    // step 7: solve L'*y = z
    cusparseScsrsv2_solve(handle, trans_Lt, m, nnz, &alpha, descr_L,
       d_csrVal, d_csrRowPtr, d_csrColInd, info_Lt,
       d_z, d_y, policy_Lt, pBuffer);

    // step 6: free resources
    cudaFree(pBuffer);
    cusparseDestroyMatDescr(descr_M);
    cusparseDestroyMatDescr(descr_L);
    cusparseDestroyCsric02Info(info_M);
    cusparseDestroyCsrsv2Info(info_L);
    cusparseDestroyCsrsv2Info(info_Lt);
    cusparseDestroy(handle);
}
