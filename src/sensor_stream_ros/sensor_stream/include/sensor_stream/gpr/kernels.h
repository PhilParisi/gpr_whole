#ifndef KERNELS_H
#define KERNELS_H

namespace gpr {
namespace kernels {

#define F_PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062f
/*!
 * \brief square exponential kernel
 * \link https://www.cs.toronto.edu/~duvenaud/cookbook/
 * \param x1 input: the first array of x vectors
 * \param x2 input: the second array of x vectors
 * \param outMat output: the generated covariance matrix
 * \param outMat_rows output: the number of rows of the output matrix
 * \param outMat_cols output: the number of cols of the output matrix
 * \param hyperParams input: hyperparams[0]:=length scale
 * \param sigma
 */
__global__ void sqExpKernel2d(float * x1, float * x2,
                           float * outMat, unsigned int outMat_rows, unsigned int outMat_cols,
                           float * hyperParams, float sigma);

/*!
 * \brief square exponential kernel
 * \link https://www.cs.toronto.edu/~duvenaud/cookbook/
 * \param x1 input: the first array of x vectors
 * \param x2 input: the second array of x vectors
 * \param outMat output: the generated covariance matrix
 * \param outMat_rows output: the number of rows of the output matrix
 * \param outMat_cols output: the number of cols of the output matrix
 * \param hyperParams input: hyperparams[0]:= fine length scale  hyperparams[1]:= course length scale
 * \param sigma
 */
__global__ void nonstationarySqExp2d(float * x1, float * x2,
                           float * outMat, unsigned int outMat_rows, unsigned int outMat_cols,
                           float * hyperParams, float sigma);


/*!
 * \brief Rational Quadratic Kernel
 * \link https://www.cs.toronto.edu/~duvenaud/cookbook/
 * \param x1 input: the first array of x vectors
 * \param x2 input: the second array of x vectors
 * \param outMat output: the generated covariance matrix
 * \param outMat_rows output: the number of rows of the output matrix
 * \param outMat_cols output: the number of cols of the output matrix
 * \param hyperParams input: hyperparams[0]:=length scale | hyperparams[1]:=alph
 * \param sigma
 */
__global__ void rqKernel2d(float * x1, float * x2,
                           float * outMat, unsigned int outMat_rows, unsigned int outMat_cols,
                           float * hyperParams, float sigma);


/*!
 * \brief Intrinssecly sparse Square Exponential Like 2d kernel
 * \link  https://www.aaai.org/ocs/index.php/IJCAI/IJCAI-09/paper/viewFile/630/840
 * \param x1 input: the first array of x vectors
 * \param x2 input: the second array of x vectors
 * \param outMat output: the generated covariance matrix
 * \param outMat_rows output: the number of rows of the output matrix
 * \param outMat_cols output: the number of cols of the output matrix
 * \param hyperParams input: hyperparams[0]:=length scale
 * \param sigma
 */
__global__ void sqExpSparse2d(float * x1, float * x2,
                           float * outMat, unsigned int outMat_rows, unsigned int outMat_cols,
                           float * hyperParams, float sigma);

}
}

#endif // KERNELS_H
