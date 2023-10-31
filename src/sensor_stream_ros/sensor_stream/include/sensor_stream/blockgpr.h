#ifndef BLOCKGPR_H
#define BLOCKGPR_H

#include "gpr.h"
#include "cudablockmat.h"
#include <thread>
#include <nvToolsExt.h>
#include "perf_timer.h"
namespace gpr {
  /*!
  * \brief Params specific to a block GPR computation
  */
  struct BlockParams{
    size_t size;
    typedef std::shared_ptr<BlockParams> ptr ;
  };

//  /*!
//   * \brief A container for all params needed for a complet BlockGPR computation/prediction
//   */
//  struct BlockGprParams{
//    RegressionParams::ptr   regression;
//    PredParams::ptr         prediction;
//    BlockParams::ptr        block;
//    /*!
//     * \brief Resets all member data pointers
//     */
//    void reset(){
//      regression.reset();
//      prediction.reset();
//      block.reset();
//    }
//    typedef std::shared_ptr<BlockParams> ptr;
//  };
}
class BlockGpr : public Gpr
{
public:
  typedef std::shared_ptr<BlockGpr> Ptr;
    BlockGpr();
    BlockGpr(BlockGpr & other);
    void addTrainingData(CudaMat<float> xTrain, CudaMat<float> yTrain, CudaMat<float> sensor_var);

    /*!
     * \brief this version of addTrainingData assumes we are using constant sensor_var which is specified in params.regression->sensor_var;
     * \param xTrain
     * \param yTrain
     */
    void addTrainingData(CudaMat<float> xTrain, CudaMat<float> yTrain);

    /*!
     * \brief generates a prediction vector with the specified parameters
     * \param x1Min lower bound of x1
     * \param x1Max upper bound of x1
     * \param x2Min lower bound of x2
     * \param x2Max upper bound of x2
     * \param x1Div number of divisions on x1
     * \param x2Div number of divisions on x2
     * \return an array of 2d points with 2 columns and x1Div*x2Div rows
     */
    CudaMat<float> genPredBlock(float x1Min, float x1Max,float x2Min, float x2Max, size_t x1Div, size_t x2Div);
    /*!
     * \brief Find the log of the determinant of the covariance matrix using the cholesky factor
     * \return
     */
    static float logDetCholBlock(CudaMat<float> cholBlock);
    float logDetChol();
    /*!
     * \brief  Compute alpha where...
     * alpha = K^-1 y
     */
    void  computeAlpha();
    float lmlBetaTerm();
//    float lml(CudaBlockMat<float> test);
    float lml();
    float derivativeLML(size_t hp_index);
    gpr::Prediction predict(CudaMat<float> predGrid);
    //gpr::Prediction predict(CudaBlockMat<float> predGrid);
    gpr::Prediction predict(float x1Min, float x1Max,float x2Min, float x2Max, size_t x1Div, size_t x2Div);
    CudaBlockMat<float> & getCholeskyFactor(){return _chol;}
    CudaBlockMat<float> & getXTrain(){return _xTrain;}
    CudaBlockMat<float> & getYTrain(){return _yTrain;}
    CudaBlockMat<float> & getAlpha(){return alpha;}
protected:
    CudaBlockMat<float> _xTrain, _yTrain, _chol,alpha;


};



#endif // BLOCKGPR_H
