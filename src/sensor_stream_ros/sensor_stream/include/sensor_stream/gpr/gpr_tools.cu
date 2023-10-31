#include "gpr_tools.h"


__global__ void likelihoodKernel(float * liklihood, float * mu_est,  float * mu_obs, float * var_est, float * var_obs, size_t rows){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<rows){
    liklihood[i] = expf(-0.5*powf(mu_est[i]-mu_obs[i],2.0f)/
                        (var_obs[i]+var_est[i]))
        /
        sqrtf(2*M_PI*(var_obs[i]+var_est[i]));
  }
}

CudaMat<float>  estimateLikelihood(CudaMat<float> mu_est,CudaMat<float> mu_obs, CudaMat<float> var_est, CudaMat<float> var_obs){
  assert(mu_est.rows()==mu_obs.rows() && mu_est.rows()==var_est.rows() && mu_est.rows()==var_obs.rows());
  //assert(mu_est.cols()==mu_obs.cols()==var_est.cols()==var_obs.cols());

  CudaMat<float> liklihood(mu_est.rows(),mu_est.cols(),dev);
  dim3 threads_per_block (32, 1, 1);
  dim3 number_of_blocks ((mu_est.rows() / threads_per_block.x) + 1, 1, 1);

  likelihoodKernel <<< number_of_blocks, threads_per_block >>>(liklihood.getRawDevPtr(),
                                                               mu_est.getRawDevPtr(),
                                                               mu_obs.getRawDevPtr(),
                                                               var_est.getRawDevPtr(),
                                                               var_obs.getRawDevPtr(),
                                                               mu_est.rows());
  cudacall(cudaDeviceSynchronize());
  return liklihood;
}
