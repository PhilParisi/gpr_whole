#include "abstractcsrmat.h"

cusparseHandle_t AbstractCsrMat::handle=NULL;

AbstractCsrMat::AbstractCsrMat()
{
    reset();
    //handle = NULL;
    //handle;
    if(handle==NULL)
      cusparseCreate(&handle);
}


AbstractCsrMat::~AbstractCsrMat(){
    //cusparseDestroy(handle);
}


void AbstractCsrMat::reset(){
    _csrParams.reset(new CsrParams_t);
    _csrParams->cols=0;
    _csrParams->rows=0;
    _csrParams->nnz=0;
    d_csrRowPtr.reset(new thrust::device_vector<int>);
    d_csrColInd.reset(new thrust::device_vector<int>);
//    d_cooRowInd.reset(new thrust::device_vector<int>);
    h_csrRowPtr.reset(new thrust::host_vector<int> );
    h_csrColInd.reset(new thrust::host_vector<int> );
//    h_cooRowInd.reset(new thrust::host_vector<int> );
    resetData();
}

void AbstractCsrMat::dev2host(){
    *h_csrRowPtr=*d_csrRowPtr;
    *h_csrColInd=*d_csrColInd;
//    *h_cooRowInd=*d_cooRowInd;
    dev2hostData();
}

void AbstractCsrMat::host2dev(){
    *d_csrRowPtr=*h_csrRowPtr;
    *d_csrColInd=*h_csrColInd;
//    *d_cooRowInd=*h_cooRowInd;
    host2devData();
}

void AbstractCsrMat::pushBackEmpty(size_t i , size_t j){
  if(i+1>_csrParams->rows){
    if(i>h_csrRowPtr->size()-2||h_csrRowPtr->size()==0){
        h_csrRowPtr->resize(i+2);
    }
    h_csrRowPtr->operator [](i+1)=_csrParams->nnz+h_csrRowPtr->operator [](0);
    _csrParams->rows=i+1;
  }
  if(j+1>_csrParams->cols){
      _csrParams->cols=j+1;
  }
}

void AbstractCsrMat::pushBackCsr(size_t i, size_t j){
    _csrParams->nnz++;
    h_csrColInd->push_back(j);
    if(i>h_csrRowPtr->size()-2||h_csrRowPtr->size()==0){
        h_csrRowPtr->resize(i+2);
        h_csrRowPtr->operator [](i)=h_csrColInd->size()-1;  // index of last element in the val array
    }
    h_csrRowPtr->operator [](i+1)=_csrParams->nnz+h_csrRowPtr->operator [](0);

    if(i+1>_csrParams->rows){
        _csrParams->rows=i+1;
    }
    if(j+1>_csrParams->cols){
        _csrParams->cols=j+1;
    }
}


void AbstractCsrMat::printHost(){
    printf ("=== SparseMat with NNZ = %i ===\n", _csrParams->nnz);
    if(_csrParams->nnz>0){
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


void AbstractCsrMat::printHostCsrRow(){
    for(size_t i = 0; i < h_csrRowPtr->size() ; i++){
        printf ("%3i ", h_csrRowPtr->operator[](i));
    }
    std::cout<< std::endl<< std::endl;
}


void AbstractCsrMat::printHostCsrCol(){
    for(size_t i = 0; i < h_csrColInd->size() ; i++){
        printf ("%3i ", h_csrColInd->operator[](i));
    }
    std::cout<< std::endl<< std::endl;
}

//void AbstractCsrMat::updateCooRow(){
//    d_cooRowInd->resize(nnz());
//    cusparsecall(cusparseXcsr2coo(getHandle(),rawCsrRowPtr(),nnz(),rows(),rawCooRowPtr(),CUSPARSE_INDEX_BASE_ZERO));
//}

