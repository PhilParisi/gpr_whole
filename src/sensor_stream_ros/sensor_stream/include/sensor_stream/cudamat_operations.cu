#include "cudamat_operations.h"


int ss::choleskyDecompose(CudaMat<float> &in,  cublasFillMode_t fillMode){
    size_t batchSize=1;
    //cusolverDnHandle_t handle = NULL;
    thrust::device_vector<int> d_infoArray;
    d_infoArray.resize(batchSize);

    float **A=(float **)malloc(batchSize*sizeof(float *));
    float **d_Aarray;
    cudacall(cudaMalloc(&d_Aarray,batchSize*sizeof(float *)));
    A[0]=in.getRawDevPtr();
    cudacall(cudaMemcpy(d_Aarray,A,batchSize*sizeof(float *),cudaMemcpyHostToDevice));

    //cusolvercall( cusolverDnCreate(&handle) );
    cusolvercall( cusolverDnSpotrfBatched(
            CudaMat<float>::getCusolverHandle(),
            fillMode,
            in.rows(),
            d_Aarray,
            in.ld(),
            thrust::raw_pointer_cast(d_infoArray.data()),
            batchSize) );

    cudaFree(d_Aarray); free(A);
    thrust::host_vector<int> out = d_infoArray;
    return out[0];
}

void ss::choleskySolve(CudaMat<float> &a, CudaMat<float> &b, cublasFillMode_t fillMode){
    //cusolverDnHandle_t handle = NULL;
    size_t batchSize=b.cols();  // since we can only do rhs size cols = 1 lets do a batch to solve multiple

    thrust::device_vector<int> d_infoArray;
    d_infoArray.resize(batchSize);

    float **B=(float **)malloc(batchSize*sizeof(float *));
    float **d_Barray;
    cudacall(cudaMalloc(&d_Barray,batchSize*sizeof(float *)));

    float **A2=(float **)malloc(batchSize*sizeof(float *));
    float **d_Aarray2;
    cudacall(cudaMalloc(&d_Aarray2,batchSize*sizeof(float *)));

    for(size_t i=0; i<batchSize; i++){
        float *temp=b.getRawDevPtr();
        B[i]=temp+i*b.ld();
        A2[i]=a.getRawDevPtr();
    }

    cudacall(cudaMemcpy(d_Aarray2,A2,batchSize*sizeof(float *),cudaMemcpyHostToDevice));
    cudacall(cudaMemcpy(d_Barray,B,batchSize*sizeof(float *),cudaMemcpyHostToDevice));


    cudacall( cudaDeviceSynchronize() );

    //cusolvercall( cusolverDnCreate(&handle) );

    cusolvercall( cusolverDnSpotrsBatched(CudaMat<float>::getCusolverHandle(),
                            fillMode,
                            a.cols(),
                            1, /* only support rhs = 1*/
                            d_Aarray2,
                            a.ld(),
                            d_Barray,
                            b.ld(),
                            thrust::raw_pointer_cast(d_infoArray.data()),
                            batchSize) );

    cudacall(cudaFree(d_Barray)); free(B);
    cudacall(cudaFree(d_Aarray2)); free(A2);
    cudacall(cudaDeviceSynchronize() );
    return;
}


void ss::choleskyInvert(CudaMat<float> &in, cublasFillMode_t fillMode){
    size_t batchSize=1;
    //cusolverDnHandle_t handle = NULL;
    thrust::device_vector<int> d_infoArray;
    d_infoArray.resize(batchSize);

    float **A=(float **)malloc(batchSize*sizeof(float *));
    float **d_Aarray;
    cudacall(cudaMalloc(&d_Aarray,batchSize*sizeof(float *)));
    A[0]=in.getRawDevPtr();
    cudacall(cudaMemcpy(d_Aarray,A,batchSize*sizeof(float *),cudaMemcpyHostToDevice));

    //cusolvercall( cusolverDnCreate(&handle) );
    cusolvercall( cusolverDnSpotrfBatched(
            CudaMat<float>::getCusolverHandle(),
            fillMode,
            in.rows(),
            d_Aarray,
            in.ld(),
            thrust::raw_pointer_cast(d_infoArray.data()),
            batchSize) );

//    in.dev2host();
//    in.printHost();
    CudaMat<float> b;
    b.initDevIdentity(in.ld());
//    b.dev2host();
//    b.printHost();
    batchSize=b.ld();  // since we can only do rhs size cols = 1 lets do a batch to solve multiple

    float **B=(float **)malloc(batchSize*sizeof(float *));
    float **d_Barray;
    cudacall(cudaMalloc(&d_Barray,batchSize*sizeof(float *)));

    float **A2=(float **)malloc(batchSize*sizeof(float *));
    float **d_Aarray2;
    cudacall(cudaMalloc(&d_Aarray2,batchSize*sizeof(float *)));

    for(size_t i=0; i<batchSize; i++){
        float *temp=b.getRawDevPtr();
        B[i]=temp+i*b.ld();
        A2[i]=in.getRawDevPtr();
    }

    cudacall(cudaMemcpy(d_Aarray2,A2,batchSize*sizeof(float *),cudaMemcpyHostToDevice));
    cudacall(cudaMemcpy(d_Barray,B,batchSize*sizeof(float *),cudaMemcpyHostToDevice));


    cudacall( cudaDeviceSynchronize() );

    cusolverDnSpotrsBatched(CudaMat<float>::getCusolverHandle(),
                            fillMode,
                            in.cols(),
                            1, /* only support rhs = 1*/
                            d_Aarray2,
                            in.ld(),
                            d_Barray,
                            b.ld(),
                            thrust::raw_pointer_cast(d_infoArray.data()),
                            batchSize);

    in = b;

    cudaFree(d_Aarray); free(A);
    cudaFree(d_Barray); free(B);
    cudaFree(d_Aarray2); free(A2);
    cudacall(cudaDeviceSynchronize() );
    return;

}

void ss::triangleSolve(CudaMat<float> &a, CudaMat<float> &b, cublasFillMode_t fillA, cublasOperation_t opA, cublasOperation_t opB){
    float alpha =1;
    if(b.isZero()){
      return;  // if b is zero the result will also be zero
    }
    assert(a.rows()==a.cols());
    assert(a.rows()==b.rows());
    switch (opB) {
    case CUBLAS_OP_N:
      cublascall(cublasStrsm(a.getHandle(),CUBLAS_SIDE_LEFT,fillA,opA,CUBLAS_DIAG_NON_UNIT,
                  b.rows(),b.cols(),&alpha,a.getRawDevPtr(),a.ld(),b.getRawDevPtr(),b.ld()));
      break;
    case CUBLAS_OP_T:
      b.transpose();
      cublascall(cublasStrsm(a.getHandle(),CUBLAS_SIDE_LEFT,fillA,opA,CUBLAS_DIAG_NON_UNIT,
                  b.rows(),b.cols(),&alpha,a.getRawDevPtr(),a.ld(),b.getRawDevPtr(),b.ld()));
      b.transpose();
      break;
    case CUBLAS_OP_C:
      throw std::logic_error("CUBLAS_OP_C not implemented for triangleSolve");
      break;
    }



}

__global__ void multDiagonalKernel(float * out, float * a,size_t ldA, size_t colsA, float * b,size_t ldB, cublasOperation_t opA, cublasOperation_t opB){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i]=0;
    size_t aIndex;
    size_t bIndex;
    if(i<ldA){
        for(unsigned j=0;j<colsA;j++){
            if(opA==CUBLAS_OP_N)
                aIndex = j*ldA+i;
            else
                aIndex = i*ldA+j;

            if(opB==CUBLAS_OP_T)
                bIndex = j*ldB+i;
            else
                bIndex = i*ldB+j;

            out[i]+=a[aIndex]*b[bIndex];
        }
    }
}

CudaMat<float> ss::multDiagonal(CudaMat<float> &a, CudaMat<float> &b,cublasOperation_t opA,cublasOperation_t opB){
    size_t len;
    if(opA==CUBLAS_OP_N)
        len = a.rows();
    else
        len = a.cols();
    CudaMat<float> out(len,1,dev);

    dim3 threads_per_block (32, 32, 1);
    dim3 number_of_blocks ((len / threads_per_block.x) + 1, 1, 1);

    multDiagonalKernel <<< number_of_blocks, threads_per_block >>> (out.getRawDevPtr(),
                                                                    a.getRawDevPtr(),a.ld(),a.cols(),
                                                                    b.getRawDevPtr(),b.ld(),
                                                                    opA,opB);
    cudacall( cudaDeviceSynchronize() );
    return out;

}

__global__ void elementWiseDivideKernel(float * a, float * b, size_t rows, size_t cols){
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t j = blockIdx.y * blockDim.y + threadIdx.y;
  size_t index = j*rows+i;//ss::idx::colMajor(i,j,rows);
  if(i<rows && j<cols){
    a[index]=a[index]/b[index];
  }
}

void ss::elementWiseDivide(CudaMat<float> & a, CudaMat<float> & b){
  assert(a.rows()==b.rows() && a.cols()==b.cols());
  dim3 threads_per_block (32, 32, 1);
  dim3 number_of_blocks ((a.rows() / threads_per_block.x) + 1, (a.cols() / threads_per_block.y) + 1, 1);
  elementWiseDivideKernel <<< number_of_blocks, threads_per_block >>>(a.getRawDevPtr(),b.getRawDevPtr(),a.rows(),a.cols());
}


__global__ void getDiagonalKernel(float * a, float * out, size_t rows){
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t mat_idx = i*rows+i;
  if(i<rows){
    out[i]=a[mat_idx];
  }
}

CudaMat<float> ss::getDiagonal(CudaMat<float> a){
  assert(a.rows()==a.cols());
  CudaMat<float> out(a.rows(),1,dev);
  dim3 threads_per_block (std::min(a.rows(), size_t(1024)), 1, 1);
  dim3 number_of_blocks ((a.rows() / threads_per_block.x) + 1, 1, 1);
  getDiagonalKernel <<< number_of_blocks, threads_per_block >>>(a.getRawDevPtr(),out.getRawDevPtr(),a.rows());
//  out.dev2host();
//  out.printHost();
  cudacall( cudaDeviceSynchronize() );
  return out;
}

__global__ void setConstantKernel(float * a, float alpha, size_t size){
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<size){
    a[i]=alpha;
  }
}

void ss::setConstant(CudaMat<float> a,float alpha){
  dim3 threads_per_block (32, 1, 1);
  dim3 number_of_blocks ((a.size() / threads_per_block.x) + 1, 1, 1);
  setConstantKernel <<< number_of_blocks, threads_per_block >>>(a.getRawDevPtr(),alpha ,a.size());

}

__global__ void traceKernel(float * x, size_t rows, float * trace){
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t mat_idx = i*rows+i;
  if(i<rows){
    //printf("adding  %f, to trace: %f at matrix index: %i rows: %i \n", x[mat_idx], trace, mat_idx, rows);
    atomicAdd(trace,x[mat_idx]);
  }
}

float ss::trace(CudaMat<float> x){
  float *tr;
    cudaMallocManaged(&tr, 4);
    *tr = 0;
  dim3 threads_per_block (std::min(x.rows(),size_t(1024)), 1, 1);
  dim3 number_of_blocks ((x.rows() / threads_per_block.x) + 1, 1, 1);
  traceKernel <<< number_of_blocks, threads_per_block >>>(x.getRawDevPtr(),x.rows(),tr);
  cudaDeviceSynchronize();
  float out = *tr;
  cudaFree(tr);
  return out;
}
