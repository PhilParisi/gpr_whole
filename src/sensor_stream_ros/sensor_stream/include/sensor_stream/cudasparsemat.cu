#include "cudasparsemat.h"

template <class T>
CudaSparseMat<T>::CudaSparseMat()
{
    _nnz=0;
//    handle = NULL;
//    cusparseCreate(&handle);
    _rows=0;
    _cols=0;
//    cusparseMatrixType_t=CUSPARSE_MATRIX_TYPE_GENERAL;
//    _baseType=CUSPARSE_INDEX_BASE_ZERO;
}

template <class T>
void CudaSparseMat<T>::reset(size_t dataSize){
    d_csrRowPtr.reset(new thrust::device_vector<int>);
    d_csrColInd.reset(new thrust::device_vector<int>);
    d_csrVal.reset(new thrust::device_vector<T>);

    h_csrRowPtr.reset(new thrust::host_vector<int> );
    h_csrColInd.reset(new thrust::host_vector<int> );
    h_csrVal.reset(new thrust::host_vector<T> );

//    size_t _rows;
//    size_t _cols;

    _dataSize=dataSize;
}

template <>
void CudaSparseMat<float>::initFromDense(CudaMat<float> in){

    this->reset();
    thrust::device_vector<int> nnzPerRow;
    nnzPerRow.resize(in.rows());
    _rows=in.rows();
    _cols=in.cols();

    int ld = in.ld(); // need to cast as int

    cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW , in.rows(),
                 in.cols() , this->getDescriptor(),
                 in.getRawDevPtr(),
                 ld, thrust::raw_pointer_cast(nnzPerRow.data()), &_nnz);

    d_csrVal->resize(_nnz);
    d_csrRowPtr->resize(in.rows()+1);
    d_csrColInd->resize(_nnz);

    cusparseSdense2csr(handle,
                       in.rows(), in.cols(),
                       this->getDescriptor(),
                       in.getRawDevPtr(),
                       ld,
                       thrust::raw_pointer_cast(nnzPerRow.data()),
                       thrust::raw_pointer_cast(d_csrVal->data()),
                       thrust::raw_pointer_cast(d_csrRowPtr->data()),
                       thrust::raw_pointer_cast(d_csrColInd->data()));
}

template <class T>
void CudaSparseMat<T>::initDev(){
    d_csrRowPtr->reserve(_dataSize);
    d_csrColInd->reserve(_dataSize);
    d_csrVal->reserve(_dataSize);
}

template <class T>
void CudaSparseMat<T>::initHost(){
    h_csrRowPtr->reserve(_dataSize);
    h_csrColInd->reserve(_dataSize);
    h_csrVal->reserve(_dataSize);
}

template <class T>
void CudaSparseMat<T>::host2dev(){
    *d_csrRowPtr=*h_csrRowPtr;
    *d_csrColInd=*h_csrColInd;
    *d_csrVal=*h_csrVal;
}

template <class T>
void CudaSparseMat<T>::dev2host(){
    *h_csrRowPtr=*d_csrRowPtr;
    *h_csrColInd=*d_csrColInd;
    *h_csrVal=*d_csrVal;
}

template <class T>
void CudaSparseMat<T>::pushBackHost(T val, unsigned int i , unsigned int j){
    _nnz++;
    h_csrVal->push_back(val);
    h_csrColInd->push_back(j);
    if(i>h_csrRowPtr->size()-2||h_csrRowPtr->size()==0){
        h_csrRowPtr->resize(i+2);
        h_csrRowPtr->operator [](i)=h_csrVal->size()-1;  // index of last element in the val array
    }
    h_csrRowPtr->operator [](i+1)=_nnz+h_csrRowPtr->operator [](0);

    if(i+1>_rows){
        _rows=i+1;
    }
    if(j+1>_cols){
        _cols=j+1;
    }
}

template <class T>
void CudaSparseMat<T>::printHost(){
    printf ("=== SparseMat with NNZ = %i ===\n", _nnz);
    if(_nnz>0){
        for(size_t i = 0; i<rows(); i++){
            size_t start = h_csrRowPtr->operator[](i);
            size_t end = h_csrRowPtr->operator[](i+1);

            for(size_t j = 0; j<cols(); j++){
                int nextCol = h_csrColInd->operator[](start);
                if(nextCol==j&&start<end){
                    printHostData(start);
                    start++;
                }else
                    printZero();
            }
            std::cout<<std::endl;
        }
    std::cout<<std::endl;
    }else
        printf("Empty Matrix \n");
}

template <class T>
void CudaSparseMat<T>::printHostData(int i){
    printf ("%10.4f ", h_csrVal->operator[](i));
}

template <class T>
void CudaSparseMat<T>::printZero(){
    printf ("    0      ");
}

template <class T>
void CudaSparseMat<T>::printHostRawData(){
    for(size_t i = 0; i < h_csrVal->size() ; i++){
        printHostData(i);
    }
    std::cout<< std::endl<< std::endl;
}

template <class T>
void CudaSparseMat<T>::printHostCsrRow(){
    for(size_t i = 0; i < h_csrRowPtr->size() ; i++){
        printf ("%3i ", h_csrRowPtr->operator[](i));
    }
    std::cout<< std::endl<< std::endl;
}

template <class T>
void CudaSparseMat<T>::printHostCsrCol(){
    for(size_t i = 0; i < h_csrColInd->size() ; i++){
        printf ("%3i ", h_csrColInd->operator[](i));
    }
    std::cout<< std::endl<< std::endl;
}


template <class T>
T CudaSparseMat<T>::val(size_t i, size_t j){
    size_t rowBegin = h_csrRowPtr->operator [](i);
    size_t rowEnd;
    if(i+1 < h_csrRowPtr->size()){
        rowEnd=h_csrRowPtr->operator [](i+1);
    }else{
        rowEnd=h_csrVal->size()-1;
    }

    size_t valIdx;
    for(valIdx=rowBegin; valIdx==j ; valIdx++){
        if(valIdx>=rowEnd){
            return T();  // value not found must be a zero
        }
    }
    return h_csrVal->operator [](valIdx);
////    size_t rowIdx=h_csrRowPtr->operator [](i);
////    size_t valIdx=rowIdx;
////    for(valIdx=rowIdx; valIdx==j ; valIdx++){
////        if(valIdx>=h_csrVal->size()){
////            return T();  // value not found must be a zero
////        }
////    }
////    return h_csrVal->operator [](valIdcusparseMatDescr_tx);
}

template <class T>
T * CudaSparseMat<T>::rawValPtr(){
    return thrust::raw_pointer_cast(d_csrVal->data());
}

template <class T>
int * CudaSparseMat<T>::rawCsrRowPtr(){
    return thrust::raw_pointer_cast(d_csrRowPtr->data());
}

template <class T>
int * CudaSparseMat<T>::rawCsrColPtr(){
    return thrust::raw_pointer_cast(d_csrColInd->data());
}

template <class T>
size_t CudaSparseMat<T>::nnz(){
    return _nnz;
}

__global__ void offsetRowsKer(int * csrRowPtr, size_t rows, size_t offset){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t vecSize = rows+1;
    if(i<vecSize){
        csrRowPtr[i]+=offset;
    }
}

void offsetRows(CudaSparseMat<float> & x, size_t offset){
    dim3 threads_per_block (32, 1, 1);
    dim3 number_of_blocks ((x.rows() / threads_per_block.x) + 1, 1, 1);
    offsetRowsKer <<< number_of_blocks, threads_per_block >>> (x.rawCsrRowPtr(),x.rows(),offset);
}

template <>
void CudaSparseMat<float>::addRows(CudaSparseMat<float> newRows,cusparseOperation_t transNewRows){
    if(transNewRows==CUSPARSE_OPERATION_NON_TRANSPOSE){
        offsetRows(newRows,this->nnz());
        d_csrRowPtr->pop_back();
        d_csrRowPtr->insert(d_csrRowPtr->end(),newRows.d_csrRowPtr->begin(),newRows.d_csrRowPtr->end());
        d_csrVal->insert(d_csrVal->end(),newRows.d_csrVal->begin(),newRows.d_csrVal->end());
        d_csrColInd->insert(d_csrColInd->end(),newRows.d_csrColInd->begin(),newRows.d_csrColInd->end());

        _rows+=newRows.rows();
        _cols=std::max(newRows.cols(),this->cols());
        _nnz+=newRows.nnz();
    }
}

template <>
void CudaSparseMat<float>::transposeDev(){
    std::shared_ptr<thrust::device_vector<float> > val;
    std::shared_ptr<thrust::device_vector<int> > newRowPtr;
    std::shared_ptr<thrust::device_vector<int> > newColInd;
    newRowPtr.reset(new thrust::device_vector<int>);
    newColInd.reset(new thrust::device_vector<int>);
    val.reset(new thrust::device_vector<float>);
    val->resize(this->nnz());
    newColInd->resize(this->nnz());
    newRowPtr->resize(this->cols()+1);

//    cusparsecall(cusparseCsr2cscEx2(this->handle, this->rows(), this->cols(), this->nnz(),
//                      rawValPtr(),rawCsrRowPtr(),
//                      rawCsrColPtr(), thrust::raw_pointer_cast(val->data()),
//                      thrust::raw_pointer_cast(newColInd->data()), thrust::raw_pointer_cast(newRowPtr->data()),
//                      CUSPARSE_ACTION_NUMERIC,
//                      CUSPARSE_INDEX_BASE_ZERO));
    d_csrRowPtr=newRowPtr;
    d_csrColInd=newColInd;
    d_csrVal=val;
    int oldRow=_rows;
    _rows=_cols;
    _cols=oldRow;
    cudacall(cudaDeviceSynchronize() );

}

template <class T>
cusparseMatDescr_t  CudaSparseMat<T>::getDescriptor(){
    cusparseMatDescr_t descrA = 0;
    cusparsecall(cusparseCreateMatDescr(&descrA));
//    cusparsecall(cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO));
//    cusparsecall(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
//    cusparsecall(cusparseSetMatFillMode(descrA, CUSPARSE_FILL_MODE_LOWER));
//    cusparsecall(cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT));
    return descrA;
}


// declare valid types
template class CudaSparseMat<float>;
template class CudaSparseMat<double>;
//template class CudaSparseMat<CudaMat<float> >;
//template class CudaSparseMat<CudaMat<float> >;

