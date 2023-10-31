#include "sparsemat_operations.h"


void ss::choleskyDecompose(CudaSparseMat<float> in){

    int pBufferSize_M;
    csric02Info_t info_M  = 0;
    void *pBuffer = 0;
    int numerical_zero;
    cusparseStatus_t status;
    const cusparseSolvePolicy_t policy_M  = CUSPARSE_SOLVE_POLICY_NO_LEVEL;

    cusparsecall(cusparseCreateCsric02Info(&info_M));


    cusparsecall(cusparseScsric02_bufferSize(in.getHandle(), in.rows(), in.nnz(),
        in.getDescriptor(), in.rawValPtr(), in.rawCsrRowPtr(), in.rawCsrColPtr(), info_M, &pBufferSize_M));

    cudacall(cudaMalloc((void**)&pBuffer, pBufferSize_M));


    cusparsecall(cusparseScsric02_bufferSize(in.getHandle(), in.rows(), in.nnz(),
        in.getDescriptor(), in.rawValPtr(), in.rawCsrRowPtr(), in.rawCsrColPtr(), info_M, &pBufferSize_M));

    cusparsecall(cusparseScsric02_analysis(in.getHandle(), in.rows(), in.nnz(), in.getDescriptor(),
        in.rawValPtr(), in.rawCsrRowPtr(), in.rawCsrColPtr(), info_M,
        policy_M, pBuffer));

    cusparsecall(cusparseScsric02(in.getHandle(), in.rows(), in.nnz(), in.getDescriptor(),
        in.rawValPtr(), in.rawCsrRowPtr(), in.rawCsrColPtr(), info_M, policy_M, pBuffer));

    status = cusparseXcsric02_zeroPivot(in.getHandle(), info_M, &numerical_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status){
       printf("L(%d,%d) is zero\n", numerical_zero, numerical_zero);
    }

    cudacall(cudaFree(pBuffer));
    cusparsecall(cusparseDestroyCsric02Info(info_M));
}


void ss::choleskySolve(CudaSparseMat<float> & a, CudaMat<float> & b,bool transA){
    solve(a,b,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_SOLVE_POLICY_NO_LEVEL);
    solve(a,b,CUSPARSE_OPERATION_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_SOLVE_POLICY_USE_LEVEL);
}

void ss::solve(CudaSparseMat<float> & a, CudaMat<float> &  b,
               cusparseOperation_t    opA, cusparseOperation_t opB,
               cusparseSolvePolicy_t policy,
               float alpha){
    size_t bufferSize = 0;
    void *pBuffer = 0;
    csrsm2Info_t  info  = 0;
    cusparseCreateCsrsm2Info(&info);
    int nrhs = 0;
    if(opB==CUSPARSE_OPERATION_NON_TRANSPOSE)
        nrhs=b.cols();
    else
        nrhs=b.rows();
        //nrhs=b.cols();


    cudacall(cudaDeviceSynchronize() );
    cusparsecall(cusparseScsrsm2_bufferSizeExt(a.getHandle(),0,opA,opB,a.rows(),nrhs,a.nnz(),&alpha,
                                  a.getDescriptor(),
                                  a.rawValPtr(),a.rawCsrRowPtr(),a.rawCsrColPtr(),
                                  b.getRawDevPtr(),b.ld(),info,policy,&bufferSize));

    cudacall(cudaMalloc((void**)&pBuffer, bufferSize));
    cusparsecall(cusparseScsrsm2_analysis(a.getHandle(),0,opA,opB,a.rows(),nrhs,a.nnz(),&alpha,
                             a.getDescriptor(),
                             a.rawValPtr(),a.rawCsrRowPtr(),a.rawCsrColPtr(),
                             b.getRawDevPtr(),b.ld(),info,policy,pBuffer));
//    printf("a rows: %i cols: %i \n",a.rows(),a.cols());
//    a.dev2host();
//    a.printHost();
//    printf("b \n");
//    b.dev2host();
//    b.printHost();
    cusparsecall(cusparseScsrsm2_solve(a.getHandle(),0,opA,opB,a.rows(),nrhs,a.nnz(),&alpha,
                          a.getDescriptor(),
                          a.rawValPtr(),a.rawCsrRowPtr(),a.rawCsrColPtr(),
                          b.getRawDevPtr(),b.ld(),
                          info,policy, pBuffer));
    //printf("4 \n");
    cudacall(cudaFree(pBuffer));
}
// note to self,  thread funcitons don't like reference parameters
void nonTransSolveThread(CudaBlockMat<float> A, CudaBlockMat<float> B ,size_t row, size_t col, size_t j,cusparseOperation_t opB){
  cublasOperation_t cublas_opB;
  switch (opB) {
  case CUSPARSE_OPERATION_NON_TRANSPOSE:
    cublas_opB = CUBLAS_OP_N;
    break;
  case CUSPARSE_OPERATION_TRANSPOSE:
    cublas_opB = CUBLAS_OP_T;
    break;
  case CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE:
    throw std::logic_error("CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE not implemented for transSolveThread");
    break;
  }
    CudaMat<float> bBlock = B.getBlock(col,0,opB);
    if(bBlock.size()>0){ /// \todo don't do this multiplication if bBlock is zeros
        CudaMat<float> product;
        if(bBlock.nnzCached()>0){  /// \todo remove this check and properly sparsify
            product = mult(A.getBlock(j),bBlock,CUBLAS_OP_N,cublas_opB);
            //add(B.getBlock(row,0), product,-1);  // subtract A(i,j),from B(j,i)
            addTrans(B.getBlock(row,0,opB),
                     product,  // invert the actuall block
                     -1,CUBLAS_OP_N,cublas_opB);
        }
    }
}


void nonTransSolve(CudaBlockMat<float> A, CudaBlockMat<float> B, cusparseOperation_t opB){

  cublasOperation_t op_b_block;
  switch(opB) {
  case CUSPARSE_OPERATION_NON_TRANSPOSE:
    op_b_block=CUBLAS_OP_N;
    break;
  case CUSPARSE_OPERATION_TRANSPOSE:
    op_b_block=CUBLAS_OP_T;
    break;
  }
    CudaMat<float> bBlock = B.getBlock(0,0,opB);
    if(bBlock.size()>0){
        ss::triangleSolve(A.getBlock(0),B.getBlock(0,0),CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,op_b_block);
    }

    for(size_t i = 1; i<A.rows(); i++){
        std::vector<std::thread> threads;
        size_t start =A.HostCsrRow(i);  // set start of row
        size_t end = A.HostCsrRow(i+1); // set end of row
        for(size_t j = start; j<end; j++){
            size_t row = i;
            size_t col = A.HostCsrCol(j);
            if(col<row){
              nonTransSolveThread(A,B,row, col, j, opB);
            }
        }
        ss::triangleSolve(A.getBlock(end-1),B.getBlock(i,0,opB),CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,op_b_block);
    }
}

void transSolveThread(CudaBlockMat<float> A, CudaBlockMat<float> B ,size_t i, size_t j, cusparseOperation_t opB){
    cublasOperation_t cublas_opB;
    switch (opB) {
    case CUSPARSE_OPERATION_NON_TRANSPOSE:
      cublas_opB = CUBLAS_OP_N;
      break;
    case CUSPARSE_OPERATION_TRANSPOSE:
      cublas_opB = CUBLAS_OP_T;
      break;
    case CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE:
      throw std::logic_error("CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE not implemented for transSolveThread");
      break;
    }
    CudaMat<float> bBlock=B.getBlock(j,0,opB);
    CudaMat<float> aBlock=A.getBlock(j,i);//we took the transpose of a so switch i and j
    if(bBlock.size()>0&&aBlock.size()>0){
        if(bBlock.nnzCached()>0){
            CudaMat<float> product = mult(aBlock,bBlock,CUBLAS_OP_T,cublas_opB);
            addTrans(B.getBlock(i,0,opB),
                     product,  // invert the actuall block
                     -1,CUBLAS_OP_N,cublas_opB);
        }

    }
}

void transSolve(CudaBlockMat<float> A, CudaBlockMat<float> B, cusparseOperation_t opB){
  cublasOperation_t op_b_block;
  switch(opB) {
  case CUSPARSE_OPERATION_NON_TRANSPOSE:
    op_b_block=CUBLAS_OP_N;
    break;
  case CUSPARSE_OPERATION_TRANSPOSE:
    op_b_block=CUBLAS_OP_T;
    break;
  }
    size_t i=A.rows()-1;
        CudaMat<float> bBlock = B.getBlock(i,0,opB);
        CudaMat<float> aBlock = A.getBlock(i,i);
        if(bBlock.size()>0){
            ss::triangleSolve(aBlock,bBlock,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,op_b_block); // need to transpose aBlock
        }
        //i--;
        for(size_t I=i;I>0;I--){
            i=I-1;
            std::vector<std::thread> threads;
            for(size_t j=i; j<A.rows(); j++ ){
                if(j>i){
                    transSolveThread(A,B,i,j,opB);
                }
            }
            ss::triangleSolve(A.getBlock(i,i),B.getBlock(i,0,opB),CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,op_b_block);
        }
}

void ss::ltSolve(CudaBlockMat<float> A, CudaBlockMat<float> B,
                         cusparseOperation_t opA,
                         cusparseOperation_t opB){
    if(opA==CUSPARSE_OPERATION_NON_TRANSPOSE){
        nonTransSolve(A,B,opB);
    }
    else{
        transSolve(A,B,opB);
    }

}



void ss::ltSolveMat(CudaBlockMat<float> L, CudaBlockMat<float> & B){

  // error checks
  if(L.rows()!=B.rows()){
    throw std::out_of_range("size missmatch! ltSolveMat requires L and B have the same number of rows");
  }
  if(L.getBlockParam().rows!=B.getBlockParam().rows){
    throw std::out_of_range("size missmatch! ltSolveMat requires the blocks of L and B have the same number of rows");
  }


  CudaBlockMatCSC<float> x;
  x.setBlockParam(B.getBlockParam());
  for (int b_col = 0 ;b_col<B.cols(); b_col++) {   // per column in x
    for (int l_row = 0 ; l_row<L.rows(); l_row++){  // per row in L

      CudaMat<float> out_block = B.getBlock(l_row,b_col);
      for (int l_col = 0; l_col<l_row ; l_col++){  // per column in L
        CudaMat<float> product = mult(L.getBlock(l_row,l_col) , x.getBlock(l_col,b_col));
        add(out_block, product, -1);
      }

      // case for l_row == l_col
      ss::triangleSolve(L.getBlock(l_row,l_row),out_block,CUBLAS_FILL_MODE_LOWER);

      // add the block to x;
      x.pushBackHost(out_block,l_row,b_col);
    }
  }
  B = x.toCSR();
}

void ss::utSolveMat(CudaBlockMat<float> U, CudaBlockMat<float> & B, cublasOperation_t opU_blocks){
  // error checks
  if(U.rows()!=B.rows()){
    throw std::out_of_range("size missmatch! ltSolveMat requires L and B have the same number of rows");
  }
  if(U.getBlockParam().rows!=B.getBlockParam().rows){
    throw std::out_of_range("size missmatch! ltSolveMat requires the blocks of L and B have the same number of rows");
  }


  CudaBlockMatCSC<float> x;
  x.setBlockParam(B.getBlockParam());
  for (int b_col = 0 ;b_col<B.cols(); b_col++) {   // per column in x
    std::vector<CudaMat<float> > x_column_vect;  // create a dense vector so we have random access so we can do this backwards
    x_column_vect.resize(U.rows());
    for (int u_row = U.rows()-1 ; u_row>=0; u_row--){  // per row in L
      CudaMat<float> out_block = B.getBlock(u_row,b_col);
      for (int u_col = u_row+1; u_col<U.cols() ; u_col++){  // per column in L
        CudaMat<float> product = mult(U.getBlock(u_row,u_col) , x_column_vect[u_col],opU_blocks);
        add(out_block, product, -1);
      }
      // case for l_row == l_col
      ss::triangleSolve(U.getBlock(u_row,u_row),out_block,CUBLAS_FILL_MODE_LOWER,opU_blocks);

      // add the block to x;
      x_column_vect[u_row]=out_block;
      //x.pushBackHost(out_block,U.rows()-1-u_row,b_col);
    }
    for (int u_row = 0 ; u_row<U.rows(); u_row++){
      x.pushBackHost(x_column_vect[u_row],u_row,b_col);
    }
  }
  B = x.toCSR();
}


void ss::choleskySolveMat(CudaBlockMat<float> L, CudaBlockMat<float> &B){
  ss::ltSolveMat(L,B);
  ss::utSolveMat(L.transposeHost(),B,CUBLAS_OP_T);
}

void ss::choleskySolve(CudaBlockMat<float> L, CudaBlockMat<float> B, cusparseOperation_t opB){
    ss::ltSolve(L,B,CUSPARSE_OPERATION_NON_TRANSPOSE,opB);
    ss::ltSolve(L,B,CUSPARSE_OPERATION_TRANSPOSE,opB);
}

///
/// \todo properly sparsify the input matrix and remove the nnz check
///
CudaMat<float> ss::singleColSquare(CudaBlockMat<float> X){
    CudaMat<float> out(X.getBlockParam().rows,X.getBlockParam().cols,dev);
    for(size_t i = 0; i<X.nnz(); i++){
//        if(X.getBlock(i,0).nnz(.0001)>0){
//            printf("xBlock \n");
//            X.getBlock(i).dev2host();
//            X.getBlock(i).printHost();
            CudaMat<float> product = mult(X.getBlock(i),X.getBlock(i),CUBLAS_OP_T,CUBLAS_OP_N);
//            printf("product \n");
//            product.dev2host();
//            product.printHost();
            out+=product;
//        }
    }
    return out;
}

CudaMat<float> ss::singleRowMult(CudaBlockMat<float> A, CudaBlockMat<float> B, size_t A_row, size_t B_col,
                        cusparseOperation_t opA,cusparseOperation_t opB){
  cublasOperation_t cublas_opB;
  switch (opB) {
  case CUSPARSE_OPERATION_NON_TRANSPOSE:
    cublas_opB = CUBLAS_OP_N;
    break;
  case CUSPARSE_OPERATION_TRANSPOSE:
    cublas_opB = CUBLAS_OP_T;
    break;
  case CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE:
    throw std::logic_error("CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE not implemented for singleRowMult");
    break;
  }

  cublasOperation_t cublas_opA;
  switch (opA) {
  case CUSPARSE_OPERATION_NON_TRANSPOSE:
    cublas_opA = CUBLAS_OP_N;
    break;
  case CUSPARSE_OPERATION_TRANSPOSE:
    cublas_opA = CUBLAS_OP_T;
    break;
  case CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE:
    throw std::logic_error("CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE not implemented for singleRowMult");
    break;
  }
  size_t cols_in_a = A.cols(opA);
  CudaMat<float> out;//(A.getBlockParam().getRows(cublas_opA),B.getBlockParam().getCols(cublas_opB),dev);
  out.setZero();
  for(size_t A_col = 0; A_col<A.cols(opA); A_col++){
    CudaMat<float> A_block = A.getBlock(A_row,A_col,opA);
    CudaMat<float> B_block = B.getBlock(A_col,B_col,opB);
    CudaMat<float> product = mult(A.getBlock(A_row,A_col,opA),
                                  B.getBlock(A_col,B_col,opB),
                                  cublas_opA,cublas_opB);
    out+=product;
  }
  return out;

}

CudaBlockMat<float> ss::mult(CudaBlockMat<float> A, CudaBlockMat<float> B,
                    cusparseOperation_t opA,
                    cusparseOperation_t opB){

  cublasOperation_t cublas_opB;
  switch (opB) {
  case CUSPARSE_OPERATION_NON_TRANSPOSE:
    cublas_opB = CUBLAS_OP_N;
    break;
  case CUSPARSE_OPERATION_TRANSPOSE:
    cublas_opB = CUBLAS_OP_T;
    break;
  case CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE:
    throw std::logic_error("CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE not implemented for singleRowMult");
    break;
  }

  cublasOperation_t cublas_opA;
  switch (opA) {
  case CUSPARSE_OPERATION_NON_TRANSPOSE:
    cublas_opA = CUBLAS_OP_N;
    break;
  case CUSPARSE_OPERATION_TRANSPOSE:
    cublas_opA = CUBLAS_OP_T;
    break;
  case CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE:
    throw std::logic_error("CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE not implemented for singleRowMult");
    break;
  }
  CudaBlockMat<float> out;
  denseParam_t dim(A.getBlockParam().getRows(cublas_opA),B.getBlockParam().getCols(cublas_opB));
  out.setBlockParam(dim);
  CudaMat<float> current_block;
  for(size_t out_row = 0; out_row < A.rows(opA); out_row++){
    for(size_t out_col = 0; out_col < B.cols(opB); out_col++){
      current_block = singleRowMult(A,B,out_row,out_col,opA,opB);
      out.pushBackHost(current_block,out_row,out_col);
    }
  }
  return out;
}

void ss::add(CudaBlockMat<float> & A, CudaBlockMat<float> B, float alpha){
  CudaBlockMat<float> A_output;
  A_output.setBlockParam(A.getBlockParam());
  CudaMat<float> current_block;
  for(size_t out_row = 0; out_row < A.rows(); out_row++){
    for(size_t out_col = 0; out_col < A.cols(); out_col++){
      current_block =  A.getBlock(out_row,out_col);
      //current_block += B.getBlock(out_row,out_col);
      add(current_block,B.getBlock(out_row,out_col),alpha);
      A_output.pushBackHost(current_block, out_row,out_col);
    }
  }
  A=A_output;
}

CudaBlockMat<float> ss::blockMatIdentity(size_t blockmat_dim, size_t block_dim){
  CudaBlockMat<float> out;
  out.setBlockDim(block_dim);
  for(size_t i = 0 ; i<blockmat_dim ; i++){
    CudaMat<float> id_block;
    id_block.initDevIdentity(block_dim);
    out.pushBackHost(id_block,i,i);
  }
  return out;
}

float ss::trace(CudaBlockMat<float> x){
  float tr=0;
  for(size_t i = 0 ; i<x.rows() ; i++){
    tr+=ss::trace(x.getBlock(i,i));
  }
  return tr;
}
