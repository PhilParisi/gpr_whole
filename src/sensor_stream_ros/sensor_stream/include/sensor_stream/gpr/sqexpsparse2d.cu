#include "sqexpsparse2d.h"
namespace ss { namespace kernels {

//SqExpSparse2d::hp_keys_({"length_scale","process_noise"},2);

SqExpSparse2d::SqExpSparse2d()
{
  type_ = "SqExpSparse2d";
  hp_indices_["length_scale"]  = 0;
  hp_indices_["process_noise"] = 1;
  hp_names_[0]  = "length_scale";
  hp_names_[1] = "process_noise";
  hyperparams_.reset(2,1);
  hyperparams_.initHost();

//  hp_keys_.push_back("length_scale");
//  hp_keys_.push_back("process_noise");
}

__global__ void kernel(float * x1, float * x2,
                           float * outMat, unsigned int outMat_rows, unsigned int outMat_cols,
                           float * hyperParams, int * nnz, float * obs_var = nullptr, bool use_obs_var = false){

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int outIdx =  j*outMat_rows+i;
  unsigned int x1offset = outMat_rows;
  unsigned int x2offset = outMat_cols;


  if(i<outMat_rows && j<outMat_cols){
    float d = sqrtf(
                    powf(x1[i]-x2[j],2) +
                    powf(x1[i+x1offset]-x2[j+x2offset],2)
                   ); // distance

    if(d>=hyperParams[0]){  // if d >= the length scale
      outMat[outIdx]=0;
    }
    else{
      outMat[outIdx]= hyperParams[1] * (
            (2+cos(2*F_PI*d/hyperParams[0]))/3   *
            (1-d/hyperParams[0])                 +
            sin(2*F_PI*d/hyperParams[0])/(2*F_PI)
          );
      atomicAdd(nnz,1);
    }
    if(i==j&&use_obs_var){
        outMat[outIdx]+=obs_var[i];
    }
  }

}

CudaMat<float> SqExpSparse2d::genCovariance(CudaMat<float> &x1, CudaMat<float> &x2){
  CudaMat<float> out;
  out.reset(x1.rows(), x2.rows());
  out.initDev();

  int *nnz;
    cudaMallocManaged(&nnz, 4);
    *nnz = 0;

  dim3 threads_per_block (32, 32, 1);
  dim3 number_of_blocks ((out.rows() / threads_per_block.x) + 1, (out.cols() / threads_per_block.y) + 1, 1);

  kernel <<< number_of_blocks, threads_per_block >>> (x1.getRawDevPtr(),
                                         x2.getRawDevPtr(),
                                         out.getRawDevPtr(), out.rows() , out.cols(),
                                         hyperparams_.getRawDevPtr(),nnz);

  cudacall(cudaGetLastError() );
  cudacall(cudaDeviceSynchronize() );
  int nnz_h = *nnz;
  out.setNNZ(nnz_h);
  cudaFree(nnz);

  return out;
}

CudaMat<float> SqExpSparse2d::genCovarianceWithSensorNoise(CudaMat<float> &x1, CudaMat<float> obs_var){
  CudaMat<float> out;
  out.reset(x1.rows(), x1.rows());
  out.initDev();

  int *nnz;
    cudaMallocManaged(&nnz, 4);
    *nnz = 0;

  dim3 threads_per_block (32, 32, 1);
  dim3 number_of_blocks ((out.rows() / threads_per_block.x) + 1, (out.cols() / threads_per_block.y) + 1, 1);

  kernel <<< number_of_blocks, threads_per_block >>> (x1.getRawDevPtr(),
                                         x1.getRawDevPtr(),
                                         out.getRawDevPtr(), out.rows() , out.cols(),
                                         hyperparams_.getRawDevPtr(),nnz,obs_var.getRawDevPtr(),true);

  cudacall(cudaGetLastError() );
  cudacall(cudaDeviceSynchronize() );
  //cudaStreamSynchronize(0);
  int nnz_h = *nnz;
  out.setNNZ(nnz_h);
  cudaFree(nnz);

  return out;
}


__global__ void partialWRTProcessNoise(float * x1, float * x2,
                           float * outMat, unsigned int outMat_rows, unsigned int outMat_cols,
                           float * hyperParams, int * nnz){

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int outIdx =  j*outMat_rows+i;
  int x1offset = outMat_rows;
  int x2offset = outMat_cols;


  if(i<outMat_rows && j<outMat_cols){
    float d = sqrtf(
                    powf(x1[i]-x2[j],2) +
                    powf(x1[i+x1offset]-x2[j+x2offset],2)
                   ); // distance

    if(d>=hyperParams[0]){  // if d >= the length scale
      outMat[outIdx]=0;
    }
    else{
      outMat[outIdx]= (
            (2+cos(2*F_PI*d/hyperParams[0]))/3   *
            (1-d/hyperParams[0])                 +
            sin(2*F_PI*d/hyperParams[0])/(2*F_PI)
          );
      atomicAdd(nnz,1);
    }
  }
}

__global__ void partialWRTLengthScale(float * x1, float * x2,
                           float * outMat, unsigned int outMat_rows, unsigned int outMat_cols,
                           float * hyperParams, int * nnz){

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int outIdx =  j*outMat_rows+i;
  int x1offset = outMat_rows;
  int x2offset = outMat_cols;

  float l = hyperParams[0];


  if(i<outMat_rows && j<outMat_cols){
    float d = sqrtf(
                    powf(x1[i]-x2[j],2) +
                    powf(x1[i+x1offset]-x2[j+x2offset],2)
                   ); // distance

    if(d>=hyperParams[0] || d==0){  // if d >= the length scale
      outMat[outIdx]=0;
    }
    else{
      outMat[outIdx]=
            hyperParams[1] * (
            -(d*cos(2*d*F_PI/l))      /                  powf(l,2)+
            d * (2 + cos(2*d*F_PI/l)) /                  (3 * powf(l,2))+
            2 * d * (1-d/l) * F_PI * sin(2*d*F_PI/l)  /  (3 * powf(l,2))
            )

          ;
      atomicAdd(nnz,1);
    }
  }
}




CudaMat<float> SqExpSparse2d::partialDerivative(CudaMat<float> &x1, CudaMat<float> &x2, size_t hp_index){
  CudaMat<float> out;
  out.reset(x1.rows(), x1.rows());
  out.initDev();

  int *nnz;
  cudaMallocManaged(&nnz, 4);
  *nnz = 0;

  dim3 threads_per_block (32, 32, 1);
  dim3 number_of_blocks ((out.rows() / threads_per_block.x) + 1, (out.cols() / threads_per_block.y) + 1, 1);

  if(hp_index == hp_indices_["length_scale"]){
    partialWRTLengthScale <<< number_of_blocks, threads_per_block >>> (x1.getRawDevPtr(),
                                           x2.getRawDevPtr(),
                                           out.getRawDevPtr(), out.rows() , out.cols(),
                                           hyperparams_.getRawDevPtr(),nnz);
  }else if(hp_index == hp_indices_["process_noise"]){
    partialWRTProcessNoise <<< number_of_blocks, threads_per_block >>> (x1.getRawDevPtr(),
                                           x2.getRawDevPtr(),
                                           out.getRawDevPtr(), out.rows() , out.cols(),
                                           hyperparams_.getRawDevPtr(),nnz);
  }else {
    throw std::runtime_error("invalid hyperparameter index passed to SqExpSparse2d::partialDerivative");
  }

  cudacall(cudaGetLastError() );
  cudacall(cudaDeviceSynchronize() );
  //cudaStreamSynchronize(0);
  int nnz_h = *nnz;
  out.setNNZ(nnz_h);
  cudaFree(nnz);

  return out;
}



}}
