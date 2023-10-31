#ifndef CUDAMAT_OPERATIONS_H
#define CUDAMAT_OPERATIONS_H
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "cudamat.h"
#include <algorithm>

namespace ss {

/*!
 * \brief Invert a matrix using cholesky decompositon
 * \param in a CudaMat that is hermetian
 * \todo make a double version of this
 */
void choleskyInvert(CudaMat<float> & in, cublasFillMode_t fillMode = CUBLAS_FILL_MODE_UPPER);


/*!
 * \brief choleskyDecompose
 * \param in [input/output] the maitrix you want to decompose
 * \param fillMode indicates if lower or upper part is stored, the other part is used as a workspace.  cublasFillMode_t either CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 * \return if value = 0, the Cholesky factorization is successful. if value = k, the leading minor of order k is not positive definite.
 * \todo make a double version of this
 */
int choleskyDecompose(CudaMat<float> & in, cublasFillMode_t fillMode = CUBLAS_FILL_MODE_UPPER);

/*!
 * \brief choleskySolve solves Ax=B
 * \param a [input] a cholesky decomposed matrix
 * \param b [input/output]  this will take B as an input and return x
 * \param fillMode indicates if lower or upper part is stored, the other part is used as a workspace.  cublasFillMode_t either CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 */
void choleskySolve(CudaMat<float> & a, CudaMat<float> & b,  cublasFillMode_t fillMode = CUBLAS_FILL_MODE_UPPER);

/*!
 * \brief triangleSolve solves Ax=B
 * \param a [input] a trianglular matrix
 * \param b [input/output]  this will take B as an input and return x
 * \param fillMode indicates if lower or upper part is stored, the other part is used as a workspace.  cublasFillMode_t either CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
 */
void triangleSolve(CudaMat<float> & a, CudaMat<float> & b, cublasFillMode_t fillA = CUBLAS_FILL_MODE_LOWER, cublasOperation_t opA= CUBLAS_OP_N, cublasOperation_t opB = CUBLAS_OP_N);

/*!
 * \brief multDiagonal computes the diagonal element of the result of x*b
 * \param a
 * \param b
 * \return a vector of length min(a.rows(), b.cols()) representing the diagonal of the matrix
 */
CudaMat<float> multDiagonal(CudaMat<float> & a, CudaMat<float> & b, cublasOperation_t opA=CUBLAS_OP_N, cublasOperation_t opB=CUBLAS_OP_N);

/*!
 * \brief Divides the a(i,j) element by the b(i,j) element;
 * \param a [input/output] the matrix of numerators
 * \param b [output] the matrix od denominators
 */
void elementWiseDivide(CudaMat<float> & a, CudaMat<float> & b);

/*!
 * \brief gets the diagonal of a cudamat and returns a nx1 vector
 * \param a
 * \return
 */
CudaMat<float> getDiagonal(CudaMat<float> a);


void setConstant(CudaMat<float> a,float alpha);

float trace(CudaMat<float> x);

}






#endif // CUDAMAT_OPERATIONS_H
