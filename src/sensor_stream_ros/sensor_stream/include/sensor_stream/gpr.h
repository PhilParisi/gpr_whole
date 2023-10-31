#ifndef GPR_H
#define GPR_H

#include "cudamat.h"
#include "gp_kernels.cu"
#include "cudamat_operations.h"
#include "cudasparsemat.h"
#include "sparsemat_operations.h"
#include "gpr/kernels.h"
#include "gpr/abstractkernel.h"
namespace gpr {

/*!
 * \brief Describes the parameters needed to compute the the regression step
 */
struct RegressionParams{
  std::shared_ptr<ss::kernels::AbstractKernel> kernel;
  float sensor_var;
  float nnzThresh;
  typedef std::shared_ptr<RegressionParams> ptr ;
};

/*!
 * \brief Parameters used to describe a gridded prediction
 */
struct PredParams{
  std::vector<size_t> gridDiv;
  typedef  std::shared_ptr<PredParams> ptr;
};

struct GprParams{
  RegressionParams::ptr   regression;
  PredParams::ptr         prediction;
  /*!
   * \brief Resets all member data pointers
   */
  void reset(){
    regression.reset(new RegressionParams);
    prediction.reset(new PredParams);
  }
  typedef std::shared_ptr<GprParams> ptr;
};

/*!
 * \brief A container for the GPR predcition
 */
struct Prediction
{
  CudaMat<float> points; //!< the points where the prediction was made
  CudaMat<float> mu;     //!< the expected value of the solution
  CudaMat<float> sigma;  //!< the variance of the solution
  float LML;
  typedef std::shared_ptr<Prediction> ptr;
  void dev2host(){
    points.dev2host();
    mu.dev2host();
    sigma.dev2host();
  }
};

}

class Gpr
{
public:
    Gpr();
    void setTrainingData(CudaMat<float> xTrain, CudaMat<float> yTrain);
    void setPredVect(CudaMat<float> pred);
//    void addTrainingData(CudaMat<float> xTrain, CudaMat<float> yTrain);
    void genPredVect(float lower, float upper, size_t div);
    void genPredVect2d(float lower, float upper, size_t div);
    //void setKernel(void (*kernel)(float*, float*, float*, unsigned int, unsigned int,float*,float)){params.regression->kernel=kernel;}
    //void setHyperParam(CudaMat<float> hyperParam){params.regression->hyperParam=hyperParam;}
//    CudaMat<float> genCovariance(CudaMat<float> & x1, CudaMat<float> & x2,
//                                 void (*kernel)(float*, float*, float*, unsigned int, unsigned int,float*,float),
//                                 CudaMat<float> hyperParams, float sigma = 0);
//    CudaMat<float> genCovariance(CudaMat<float> & x1, CudaMat<float> & x2, float sigma = 0){
//      return genCovariance(x1,x2,
//                           params.regression->kernel,
//                           params.regression->hyperParam,
//                           sigma);
//    }
    //void setKernel(std::shared_ptr<ss::kernels::AbstractKernel> ker){kernel=ker;}
    gpr::Prediction prediction;
    gpr::GprParams params;
    std::shared_ptr<ss::kernels::AbstractKernel> getKernel(){return params.regression->kernel;}
protected:
    CudaMat<float> _xTrain, _yTrain;
    //std::shared_ptr<ss::kernels::AbstractKernel> kernel;
};


#endif // GPR_H
