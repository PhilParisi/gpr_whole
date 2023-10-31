#ifndef __EIGENMULTIVARIATENORMAL_HPP
#define __EIGENMULTIVARIATENORMAL_HPP

#include <Eigen/Dense>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <mutex>

/*
  We need a functor that can pretend it's const,
  but to be a good random number generator
  it needs mutable state.  The standard Eigen function
  Random() just calls rand(), which changes a global
  variable.
*/
namespace Eigen {
namespace internal {
template<typename Scalar>
struct scalar_normal_dist_op
{
  static thread_local boost::mt19937 rng;                        // The uniform pseudo-random algorithm
  mutable boost::normal_distribution<Scalar> norm;  // The gaussian combinator

  EIGEN_EMPTY_STRUCT_CTOR(scalar_normal_dist_op)

  template<typename Index>
  inline const Scalar operator() (Index, Index = 0) const { return norm(rng); }
};

template<typename Scalar>
thread_local boost::mt19937  scalar_normal_dist_op<Scalar>::rng;

template<typename Scalar>
struct functor_traits<scalar_normal_dist_op<Scalar> >
{ enum { Cost = 50 * NumTraits<Scalar>::MulCost, PacketAccess = false, IsRepeatable = false }; };

} // end namespace internal
/**
    Find the eigen-decomposition of the covariance matrix
    and then store it for sampling from a multi-variate normal
*/
template<typename Scalar, int Size>
class EigenMultivariateNormal
{
  //static std::mutex mtx;
  Matrix<Scalar,Size,Size> _covar;
  Matrix<Scalar,Size,Size> _transform;
  Matrix< Scalar, Size, 1> _mean;
  internal::scalar_normal_dist_op<Scalar> randN; // Gaussian functor


public:
  EigenMultivariateNormal(const Matrix<Scalar,Size,1>& mean,const Matrix<Scalar,Size,Size>& covar)
  {
    //mtx.lock();
    setMean(mean);
    setCovar(covar);
  }
  ~EigenMultivariateNormal(){
    //mtx.unlock();
  }

  void setMean(const Matrix<Scalar,Size,1>& mean) { _mean = mean; }
  void setCovar(const Matrix<Scalar,Size,Size>& covar)
  {
    _covar = covar;

      SelfAdjointEigenSolver<Matrix<Scalar,Size,Size> > eigenSolver(_covar);
      _transform = eigenSolver.eigenvectors()*eigenSolver.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal();

  }

  /// Draw nn samples from the gaussian and return them
  /// as columns in a Size by nn matrix
  Matrix<Scalar,Size,-1> samples(int nn)
  {
    return (_transform * Matrix<Scalar,Size,-1>::NullaryExpr(Size,nn,randN)).colwise() + _mean;
  }
}; // end class EigenMultivariateNormal

//template <typename Scalar,int Size>
//std::mutex EigenMultivariateNormal<Scalar, Size>::mtx;

} // end namespace Eigen
#endif
