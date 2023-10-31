#include "gpr.h"

Gpr::Gpr()
{
  params.reset();
}

void Gpr::setTrainingData(CudaMat<float> xTrain, CudaMat<float> yTrain){
    _xTrain=xTrain;
    _yTrain=yTrain;
}

void Gpr::setPredVect(CudaMat<float> pred){
    prediction.points=pred;
}

/*!
 * \brief kernal associated with Gpr::genPredVect
 * \param pred
 * \param lower
 * \param upper
 * \param div
 */
__global__ void genPredVectKernel(float * pred,
                            float lower,
                            float upper,
                            size_t div){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    pred[i]=i*((upper-lower)/float(div-1));
}

/*!
 * \brief generates a linearly spaced vector of predictuion points on the device
 * recomend using this for linspace instaid of copying because it elminimates host2dev copy
 * \param lower the lower bound
 * \param upper the upper bound
 * \param div the number of divisions
 */
void Gpr::genPredVect(float lower, float upper, size_t div){
    prediction.points.reset(div,1);
    prediction.points.initDev();
    //cudaDeviceProp prop;
    //cudaGetDeviceProperties(&prop, 1);
    dim3 threads_per_block (32, 1, 1);
    dim3 number_of_blocks ((prediction.points.size() / threads_per_block.x) + 1, 1, 1);
    genPredVectKernel <<< number_of_blocks, threads_per_block >>> (prediction.points.getRawDevPtr(),lower,upper,div);
    cudacall(cudaDeviceSynchronize());
    return;
}

__global__ void genPredVect2dKernel(float * pred,
                            float lower,
                            float upper,
                            size_t div){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i<div && j<div){
        pred[i+(j*div)]=i*((upper-lower)/float(div-1));
        pred[i+(j*div)+(div*div)]=j*((upper-lower)/float(div-1));
    }
}

void Gpr::genPredVect2d(float lower, float upper, size_t div){
    prediction.points.reset(div*div,2);
    prediction.points.initDev();
    dim3 threads_per_block (32, 1, 1);
    dim3 number_of_blocks ((div / threads_per_block.x) + 1, (div / threads_per_block.y) + 1, 1);
    genPredVect2dKernel <<< number_of_blocks, threads_per_block >>> (prediction.points.getRawDevPtr(),lower,upper,div);
    cudacall(cudaGetLastError() );
    cudacall(cudaDeviceSynchronize() );
}

/*!
 * \brief creates the covariance matricies on the device based on two cudamat vectors
 * \param x1
 * \param x2
 * \return
 */
//CudaMat<float> Gpr::genCovariance(CudaMat<float> & x1, CudaMat<float> & x2,
//                                  void (*kernel)(float*, float*, float*, unsigned int, unsigned int,float*,float),
//                                  CudaMat<float> hyperParams, float sigma){

//    if(hyperParams.size()==0){
//      throw std::runtime_error("Gpr::genCovariance called with no hyperparameters set. call Gpr::setHyperParam with a valid lenght hyperparm vector first!");
//    }
//    CudaMat<float> out;
//    out.reset(x1.rows(), x2.rows());
//    out.initDev();//    _mu =kp_x * vInv * _yTrain;

//    dim3 threads_per_block (32, 32, 1); // A 16 x 16 block threads
//    dim3 number_of_blocks ((out.rows() / threads_per_block.x) + 1, (out.cols() / threads_per_block.y) + 1, 1);

//    kernel <<< number_of_blocks, threads_per_block >>> (x1.getRawDevPtr(),
//                                           x2.getRawDevPtr(),
//                                           out.getRawDevPtr(), out.rows() , out.cols(),
//                                           hyperParams.getRawDevPtr(),sigma);

//    out.nnz(params.regression->nnzThresh);
//    cudacall(cudaGetLastError() );
//    cudacall(cudaDeviceSynchronize() );

//    return out;
//}
