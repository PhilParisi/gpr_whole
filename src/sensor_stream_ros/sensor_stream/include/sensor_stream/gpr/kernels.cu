#include "kernels.h"

__global__ void gpr::kernels::sqExpKernel2d(float * x1, float * x2,
                           float * outMat, unsigned int outMat_rows, unsigned int outMat_cols,
                           float * hyperParams, float sigma){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int outIdx = j*outMat_rows+i;
    int x1offset = outMat_rows;
    int x2offset = outMat_cols;


    if(i<outMat_rows && j<outMat_cols){
        outMat[outIdx] =  expf(-((pow(x1[i]-x2[j] , 2) +
                           pow(x1[i+x1offset]-x2[j+x2offset] , 2)))/
                            (hyperParams[0]*hyperParams[0]) ) ;
         if(i==j){
             outMat[outIdx]+=sigma*sigma;
         }
    }

}


__global__ void gpr::kernels::nonstationarySqExp2d(float * x1, float * x2,
                           float * outMat, unsigned int outMat_rows, unsigned int outMat_cols,
                           float * hyperParams, float sigma){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int outIdx = j*outMat_rows+i;
    int x1offset = outMat_rows;
    int x2offset = outMat_cols;


    if(i<outMat_rows && j<outMat_cols){
        outMat[outIdx] =  expf(-((pow(x1[i]-x2[j] , 2) + pow(x1[i+x1offset]-x2[j+x2offset] , 2)))/(hyperParams[0]*hyperParams[0]) ) +
                          expf(-((pow(x1[i]-x2[j] , 2) + pow(x1[i+x1offset]-x2[j+x2offset] , 2)))/(hyperParams[1]*hyperParams[1]) );
         if(i==j){
             outMat[outIdx]+=sigma*sigma;
         }
    }

}


// rational quadratic kernel
__global__ void gpr::kernels::rqKernel2d(float * x1, float * x2,
                           float * outMat, unsigned int outMat_rows, unsigned int outMat_cols,
                           float * hyperParams, float sigma){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int outIdx = j*outMat_rows+i;
    int x1offset = outMat_rows;
    int x2offset = outMat_cols;
    float sig=1;
    float deltaX=x1[i]-x2[j];
    float deltaY=x1[i+x1offset]-x2[j+x2offset];
    float alpha=hyperParams[1];
    float lSq=hyperParams[0]*hyperParams[0];


    if(i<outMat_rows && j<outMat_cols){
        //outMat[outIdx] =  expf(-((pow(x1[i]-x2[j] , 2) + pow(x1[i+x1offset]-x2[j+x2offset] , 2)))/hyperParams[0] ) ;
        outMat[outIdx] = sig*sig*powf(1+(deltaX*deltaX+deltaY*deltaY)/(2*alpha*lSq),-alpha);
         if(i==j){
             outMat[outIdx]+=sigma*sigma;
         }
    }

}

// rational quadratic kernel
__global__ void gpr::kernels::sqExpSparse2d(float * x1, float * x2,
                           float * outMat, unsigned int outMat_rows, unsigned int outMat_cols,
                           float * hyperParams, float sigma){

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int outIdx = j*outMat_rows+i;
  int x1offset = outMat_rows;
  int x2offset = outMat_cols;


  if(i<outMat_rows && j<outMat_cols){
    float d = sqrt(
                    pow(x1[i]-x2[j],2) +
                    pow(x1[i+x1offset]-x2[j+x2offset],2)
                   ); // distance

    if(d>=hyperParams[0]){  // if d >= the length scale
      outMat[outIdx]=0;
    }else{
      outMat[outIdx]= hyperParams[1] * (
            (2+cos(2*F_PI*d/hyperParams[0]))/3   *
            (1-d/hyperParams[0])                 +
            sin(2*F_PI*d/hyperParams[0])/(2*F_PI)
          );
    }
    if(i==j){
        outMat[outIdx]+=sigma*sigma;
    }
  }

}

