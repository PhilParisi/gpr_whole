#ifndef ABSTRACTKERNEL_H
#define ABSTRACTKERNEL_H

#include "../cudamat.h"
#include <map>
#include <string>


namespace ss { namespace kernels {

class AbstractKernel
{
public:
  AbstractKernel();

  /*!
   * \brief find the covariance of x1 and x2
   * \param x1
   * \param x2
   * \return covariance(x1,x2)
   */
  virtual CudaMat<float> genCovariance(CudaMat<float> & x1, CudaMat<float> & x2) = 0;

  /*!
   * \brief This function will get the covarance of x1 with itself AND add the obsservation variance
   * \param x1
   * \return the covariance(x1,x1)+obs_var
   */
  virtual CudaMat<float> genCovarianceWithSensorNoise(CudaMat<float> & x1,CudaMat<float> obs_var) = 0;

  /*!
   * \brief compute the partial derviative of the covariance with respect to the selected hyperparameter
   * \param x1
   * \param x2
   * \param hp_index the index of the hyperparameter you want to differentiate WRT.
   * \return
   */
  virtual CudaMat<float> partialDerivative(CudaMat<float> & x1, CudaMat<float> & x2, size_t hp_index) = 0;

  /*!
   * \brief Gets the string that describes what kind of kernel this is
   * \return a string describing the type of kernel this is
   */
  std::string getType(){return type_;}
  virtual void setHyperparams(CudaMat<float> hp){hyperparams_=hp;}
  /*!
   * \brief get a read/write reference to the hyperparam host value specified by the key
   * \param key
   * \return
   */
  float & hyperparam(std::string key);
  float & hyperparam(size_t index);
  void hyperparam2dev(){hyperparams_.host2dev();}
  void hyperparam2host(){hyperparams_.dev2host();}
  size_t numHyperparams(){return hp_indices_.size();}
  std::map<std::string, size_t> getHPIndexMap(){return hp_indices_;}
  CudaMat<float> getHyperparams(){return hyperparams_;}
  std::string getHyperparamKey(size_t index){return hp_names_[index];}
  //static std::vector<std::string> getHpKeys();

//  virtual std::vector<size_t> getHpKeys(){return hp_keys_;}
protected:
  std::string type_;
  CudaMat<float> hyperparams_;
  std::map<std::string, size_t> hp_indices_;
  std::map<size_t, std::string> hp_names_;
  //std::shared_ptr<ceres::CostFunction> cost_function_;
  //std::vector<std::string> hp_keys_;

};

}
}

#endif // ABSTRACTKERNEL_H
