#include "cudablockmat.h"

template <class T>
CudaBlockMat<T>::CudaBlockMat()
{
    _blockParams.reset(new denseParam_t);
    h_blockVal.reset(new std::vector<CudaMat<T > > );
    d_blockVal.reset(new thrust::device_vector<T*> );
    _zero.setZero();
}

template <class T>
CudaBlockMat<T> CudaBlockMat<T>::deepCopy(){
    CudaBlockMat<T> copy;
    copy.h_blockVal->resize(this->h_blockVal->size());
    for (size_t i = 0 ;  i < this->h_blockVal->size()  ;  i++) {
        copy.h_blockVal->operator[](i)=this->h_blockVal->operator[](i).deepCopy();
    }
//    *copy.d_csrRowPtr = *d_csrRowPtr;
//    *copy.d_csrColInd = *d_csrColInd;
    *copy.h_csrRowPtr = *h_csrRowPtr;
    *copy.h_csrColInd = *h_csrColInd;
    *copy._csrParams  = *_csrParams;
//    *copy.d_blockVal = *d_blockVal;

    *copy._blockParams = *_blockParams;

    return copy;
}


template  <class T>
CudaBlockMat<T> CudaBlockMat<T>::deepCopyCSR(){
  CudaBlockMat<T> copy;
  //copy.h_blockVal->resize(this->h_blockVal->size());
//    *copy.d_csrRowPtr = *d_csrRowPtr;
//    *copy.d_csrColInd = *d_csrColInd;
  *copy.h_blockVal  = *h_blockVal;
  *copy.h_csrRowPtr = *h_csrRowPtr;
  *copy.h_csrColInd = *h_csrColInd;
  *copy._csrParams  = *_csrParams;
//    *copy.d_blockVal = *d_blockVal;

  *copy._blockParams = *_blockParams;

  return copy;
}

template <class T>
void CudaBlockMat<T>::resetData(){
  _blockParams.reset(new denseParam_t);
  h_blockVal.reset(new std::vector<CudaMat<T > > );
  d_blockVal.reset(new thrust::device_vector<T*> );
  _zero.setZero();
}

template <class T>
void CudaBlockMat<T>::host2devData(){
    thrust::host_vector<T*> hostCopy;
    hostCopy.resize(h_blockVal->size());
    for(size_t i=0; i<h_blockVal->size(); i++){
        h_blockVal->operator [](i).host2dev(); // move data to host
        hostCopy[i]=h_blockVal->operator [](i).getRawDevPtr();  // assign dev pointer to hostCopy
    }
    *d_blockVal=hostCopy;

}

template <class T>
void CudaBlockMat<T>::dev2hostData(){
  for(auto block: *h_blockVal){
    block.dev2host();
  }
}

template <class T>
void CudaBlockMat<T>::pushBackHost(CudaMat<T> val, size_t i, size_t j){
  if(val.isZero()){
    pushBackEmpty(i,j);
    return;
  }
  pushBackCsr(i,j);
  if(val.getDenseParam()==getBlockParam()){
      val.setDenseParam(_blockParams);
  }else{
      throw std::out_of_range("Size missmatch in CudaBlockMat<T>::pushBackHost: input CudaMat's size does not match the block dimension");
  }
  h_blockVal->push_back(val);
}

template <class T>
void CudaBlockMat<T>::setBlockDim(size_t n){
    _blockParams->rows=n;
    _blockParams->cols=n;
}

template <class T>
CudaMat<T> &CudaBlockMat<T>::getBlock(size_t i) const{
    return h_blockVal->operator [](i);
}

template <class T>
CudaMat<T> &CudaBlockMat<T>::getBlock(size_t row, size_t col, cusparseOperation_t op){

  size_t i,j;
  switch (op) {
  case CUSPARSE_OPERATION_NON_TRANSPOSE:
    if(row >= primaryDim()  || col >= secondaryDim()){
      throw std::out_of_range("attemped to access an out of bounds value");
    }
    i=row;
    j=col;
    break;
  case CUSPARSE_OPERATION_TRANSPOSE:
    if(col >= primaryDim()  || row >= secondaryDim()){
      throw std::out_of_range("attemped to access an out of bounds value");
    }
    j=row;
    i=col;
    break;
  case CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE:
    throw std::logic_error("CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE not implemented for CudaBlockMat<T>::getBlock");
    break;

  }
    size_t rowStartIndex = HostCsrRow(i);
    size_t rowEndIndex = HostCsrRow(i+1);
    for(size_t colIter = rowStartIndex; colIter<rowEndIndex; colIter++){
        if(HostCsrCol(colIter)==j){
            return getBlock(colIter);
        }
    }
    return _zero;
}

template <class T>
void CudaBlockMat<T>::blocks2dev(){
  for(auto block: *h_blockVal){
    block.host2dev();
  }
}

template <class T>
void CudaBlockMat<T>::blocks2host(){
  for(auto block: *h_blockVal){
    block.dev2host();
  }
}

template <class T>
T CudaBlockMat<T>::getVal(size_t i,size_t j){
  size_t block_index_i = i / getBlockParam().rows;
  size_t block_index_j = j / getBlockParam().cols;
  size_t val_index_i = i % getBlockParam().rows;
  size_t val_index_j = j % getBlockParam().cols;
  if(getBlock(block_index_i,block_index_j).isZero()){
    return 0;
  }else {
    return getBlock(block_index_i,block_index_j).val(val_index_i,val_index_j);
  }

}

template <class T>
CudaMat<T> &CudaBlockMat<T>::getCreateBlock(size_t i, size_t j){

    size_t rowStartIndex = HostCsrRow(i);
    size_t rowEndIndex = HostCsrRow(i+1);
    for(size_t colIter = rowStartIndex; colIter<rowEndIndex; colIter++){
        if(HostCsrCol(colIter)>j){

        }
        if(HostCsrCol(colIter)==j){
            return getBlock(colIter);
        }
    }
}

template <class T>
CudaBlockMat<T> CudaBlockMat<T>::transposeHost(){
  CudaBlockMat<T> b;
  b.h_csrRowPtr->resize(secondaryDim()+1);
  b.h_csrColInd->resize(nnz());
  b.h_blockVal->resize(nnz());
  b.setBlockParam(getBlockParam());
  *b._csrParams=*_csrParams;
  b._csrParams->rows = _csrParams->cols;
  b._csrParams->cols = _csrParams->rows;

  //compute number of non-zero entries per column of A
  std::fill(b.h_csrRowPtr->begin(), b.h_csrRowPtr->begin() + secondaryDim(), 0);

  for (int n = 0; n < nnz(); n++){
      b.hostCsrRowPtr(hostCsrColInd(n))++;
  }

  //cumsum the nnz per column to get Bp[]
  for(int col = 0, cumsum = 0; col < secondaryDim(); col++){
      int temp  = b.hostCsrRowPtr(col);
      b.hostCsrRowPtr(col) = cumsum;
      cumsum += temp;
  }
  b.hostCsrRowPtr(secondaryDim()) = nnz();


  for(int row = 0; row < primaryDim(); row++){


      for(int jj = hostCsrRowPtr(row); jj < hostCsrRowPtr(row+1); jj++){
          int col  = hostCsrColInd(jj);
          int dest = b.hostCsrRowPtr(col);

          b.hostCsrColInd(dest) = row;

          b.h_blockVal->operator[](dest) = h_blockVal->operator[](jj);

          b.hostCsrRowPtr(col)++;
      }
  }

  for(int col = 0, last = 0; col <= secondaryDim(); col++){
    int temp  = b.hostCsrRowPtr(col);
    b.hostCsrRowPtr(col) = last;
    last    = temp;
  }
  return b;
}

template <class T>
void CudaBlockMat<T>::printHostValues(){
  for (size_t row =0 ; row < rows()*_blockParams->rows ; row++) {
    for (size_t col = 0 ; col < cols()*_blockParams->cols; col++) {
      printf ("%8.4g ", getVal(row,col));
    }
    std::cout << std::endl;
  }

}

template <class T>
void CudaBlockMat<T>::printHostValuesMathematica(){
  printf ("{\n");
  for(size_t i = 0; i < rows()*_blockParams->rows ; i++){
      printf ("   {");
      for(size_t j=0; j < cols()*_blockParams->cols ; j++){
           printf ("%15.10f", getVal(i,j));
           if(j<cols()*_blockParams->cols-1){
               printf (",");
           }
      }
      printf ("}");
      if(i<cols()*_blockParams->cols-1){
          printf (",");
      }
      std::cout<< std::endl;
  }
  printf ("}");
  std::cout<< std::endl;
}

template <class T>
void CudaBlockMat<T>::printHostData(int i){
    float numZero = getBlock(i).size() - getBlock(i).nnz(.01);
    float percentage = numZero/getBlock(i).size();
    printf ("i:%4i   ", int(percentage*100));
}

template <class T>
CudaMat<int> CudaBlockMat<T>::getSparsityMatrix(){
  CudaMat<int> out(rows(),cols(),host);
  if(_csrParams->nnz>0){
      for(size_t i = 0; i<rows(); i++){
          size_t start = h_csrRowPtr->operator[](i);
          size_t end = h_csrRowPtr->operator[](i+1);

          for(size_t j = 0; j<cols(); j++){
              int nextCol = h_csrColInd->operator[](start);
              if(nextCol==j&&start<end){
                  //printHostData(start);
                  out(i,j)=getBlock(start).nnz(1e-30f);
                  start++;
              }else
                  out(i,j)= -1;  // case for zero
          }
      }
  }else
      printf("Empty Matrix \n");
  return  out;

}

template class CudaBlockMat<float>;
