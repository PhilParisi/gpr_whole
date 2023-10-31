#ifndef SPARSEMAT_OPERATIONS_H
#define SPARSEMAT_OPERATIONS_H
#include "cudasparsemat.h"
#include "cudablockmat.h"
#include "cudablockmatcsc.h"
#include "cudamat_operations.h"
#include <cusparse_v2.h>
#include <thread>

namespace ss {
    void choleskyDecompose(CudaSparseMat<float> in);
    void choleskySolve(CudaSparseMat<float> & a, CudaMat<float> & b,bool transA =false );
    void solve(CudaSparseMat<float> & a, CudaMat<float> & b,
               cusparseOperation_t opA =  CUSPARSE_OPERATION_NON_TRANSPOSE,
               cusparseOperation_t opB =  CUSPARSE_OPERATION_NON_TRANSPOSE,
               cusparseSolvePolicy_t policy = CUSPARSE_SOLVE_POLICY_NO_LEVEL,
               float alpha = 1);
    void threadedLtSolve(CudaBlockMat<float> A, CudaBlockMat<float> B,
                          cusparseOperation_t opA =  CUSPARSE_OPERATION_NON_TRANSPOSE,
                          cusparseOperation_t opB =  CUSPARSE_OPERATION_NON_TRANSPOSE);
    void ltSolve(CudaBlockMat<float> A, CudaBlockMat<float> B,
               cusparseOperation_t opA =  CUSPARSE_OPERATION_NON_TRANSPOSE,
               cusparseOperation_t opB =  CUSPARSE_OPERATION_NON_TRANSPOSE);

    void ltSolveMat(CudaBlockMat<float> L, CudaBlockMat<float> &B);
    void utSolveMat(CudaBlockMat<float> U, CudaBlockMat<float> &B, cublasOperation_t opU_blocks);
    void choleskySolveMat(CudaBlockMat<float> L, CudaBlockMat<float> &B);
    /*!
     * \brief choleskySolve solves the system Ax=B where
     * L is the lower triangular cholesky factor L*Transpose(L) = A
     * \param L [input]
     * \param B [input/output]
     */
    void choleskySolve(CudaBlockMat<float> L, CudaBlockMat<float> B, cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE);

    CudaMat<float> singleColSquare(CudaBlockMat<float> X);

    CudaMat<float> singleRowMult(CudaBlockMat<float> A, CudaBlockMat<float> B, size_t A_row,size_t B_col,
                             cusparseOperation_t opA =  CUSPARSE_OPERATION_NON_TRANSPOSE,
                             cusparseOperation_t opB =  CUSPARSE_OPERATION_NON_TRANSPOSE);

    CudaBlockMat<float> mult(CudaBlockMat<float> A, CudaBlockMat<float> B,
                             cusparseOperation_t opA =  CUSPARSE_OPERATION_NON_TRANSPOSE,
                             cusparseOperation_t opB =  CUSPARSE_OPERATION_NON_TRANSPOSE);

    void add(CudaBlockMat<float> & A, CudaBlockMat<float> B, float alpha = 1);

    CudaBlockMat<float> blockMatIdentity(size_t blockmat_dim, size_t block_dim);

    float trace(CudaBlockMat<float> x);

}



#endif // SPARSEMAT_OPERATIONS_H
