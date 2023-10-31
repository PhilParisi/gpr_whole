#include "blockgpr.h"

BlockGpr::BlockGpr()
{
  _xTrain.reset();
  _yTrain.reset();
  params.reset();
  params.regression->nnzThresh=0.0001f;
}

BlockGpr::BlockGpr(BlockGpr & other){
  _chol  =  other._chol.deepCopyCSR();
  _xTrain=  other._xTrain.deepCopyCSR();
  _yTrain=  other._yTrain.deepCopyCSR();
  params=other.params;
  params.regression->nnzThresh=other.params.regression->nnzThresh;
}


void BlockGpr::addTrainingData(CudaMat<float> xTrain, CudaMat<float> yTrain, CudaMat<float> sensor_var){
  nvtxRangePush(__FUNCTION__);
  nvtxMark("add trainging data...");

    if(_chol.rows()>0){
        // generate new part of covariance matrix
        CudaBlockMat<float> s12;
        s12.setBlockParam(_chol.getBlockParam());
        nvtxRangePushA("generate s12 block");
        for(size_t i = 0; i < _xTrain.rows(); i++){
            CudaMat<float> s12block = getKernel()->genCovariance(_xTrain.getBlock(i,0),xTrain);//genCovariance(_xTrain.getBlock(i,0),xTrain);
            s12.pushBackHost(s12block,i,0);
            //s12.pushBackHost(results[i],i,0);
        }
        nvtxRangePop();

        // calculate s12 by solving
        nvtxRangePushA("Updating Chol");
        ss::ltSolve(_chol,s12);

        CudaMat<float> s12_squared = ss::singleColSquare(s12);

        CudaMat<float> s22 = getKernel()->genCovarianceWithSensorNoise(xTrain,sensor_var);//genCovariance(xTrain,xTrain,params.regression->sigma2e);

        s22-=s12_squared;

        ss::choleskyDecompose(s22);

        nvtxRangePop();
        nvtxRangePushA("Adding New Rows");
        // add the results to the _chol matrix
        size_t newRowIdx = _chol.cols();
        for(size_t j = 0; j<_chol.cols(); j++){
            if(s12.getBlock(j,0).size()>0){
                if(s12.getBlock(j,0).nnzCached()>0){  // only push back if nonzero vals
                    s12.getBlock(j,0).transpose();
                    _chol.pushBackHost(s12.getBlock(j,0),newRowIdx,j);
                }
            }
        }



        _chol.pushBackHost(s22,newRowIdx,newRowIdx);
        nvtxRangePop();
        //std::cout << "NNZ _chol: " << _chol.nnz() << std::endl;
        //_chol.printHost();
        //_chol.getSparsityMatrix().printHost();



        _xTrain.setBlockParam(xTrain.getDenseParam());
        _xTrain.pushBackHost(xTrain,_xTrain.rows(),0);
        _yTrain.setBlockParam(yTrain.getDenseParam());
        _yTrain.pushBackHost(yTrain,_yTrain.rows(),0);
    } else{  // case for initial update
        _xTrain.setBlockParam(xTrain.getDenseParam());
        _xTrain.pushBackHost(xTrain,_xTrain.rows(),0);
        _yTrain.setBlockParam(yTrain.getDenseParam());
        _yTrain.pushBackHost(yTrain,_yTrain.rows(),0);

        CudaMat<float> initialChol= getKernel()->genCovarianceWithSensorNoise(xTrain,sensor_var);//genCovariance(xTrain,xTrain, params.regression->sigma2e);
        ss::choleskyDecompose(initialChol);
        _chol.setBlockParam(initialChol.getDenseParam());
        _chol.pushBackHost(initialChol,0,0);
    }
    nvtxRangePop();

}

void BlockGpr::addTrainingData(CudaMat<float> xTrain, CudaMat<float> yTrain){
  CudaMat<float> sensor_var(xTrain.rows(),xTrain.cols(),dev);
  ss::setConstant(sensor_var,params.regression->sensor_var);
  addTrainingData(xTrain,yTrain,sensor_var);
}

__global__ void genPredBlock2dKernel(float * pred,
                            float x1Min, float x1Max,float x2Min, float x2Max, size_t x1Div, size_t x2Div){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int ld = x1Div*x2Div;
    if(i<x1Div && j<x2Div){
      if(x1Div-1 == 0){
        pred[i+(j*x1Div)]=x1Min;
      }else{
        pred[i+(j*x1Div)]=i*((x1Max-x1Min)/float(x1Div-1))+x1Min;
      }

      if(x2Div-1 == 0){
        pred[i+(j*x1Div)+(ld)]=x2Min;
      }else{
        pred[i+(j*x1Div)+(ld)]=j*((x2Max-x2Min)/float(x2Div-1))+x2Min;
      }
    }
}

CudaMat<float> BlockGpr::genPredBlock(float x1Min, float x1Max,float x2Min, float x2Max, size_t x1Div, size_t x2Div){
    CudaMat<float> out;
    out.reset(x1Div*x2Div,2);
    out.initDev();
    dim3 threads_per_block (1024, 1, 1);
    dim3 number_of_blocks ((x1Div / threads_per_block.x) + 1, (x2Div / threads_per_block.y) + 1, 1);
    genPredBlock2dKernel <<< number_of_blocks, threads_per_block >>> (out.getRawDevPtr(), x1Min,  x1Max, x2Min,  x2Max,  x1Div,  x2Div);
    cudacall(cudaGetLastError() );
    cudacall(cudaDeviceSynchronize() );

    return out;
}


__global__ void logDetCholKernel(float * x, size_t ldx, size_t cols , float * det){
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t index = i*ldx+i;
  if(i<ldx && i<cols){
    //printf("log of index %i: %f ", index, logf(x[index]));
    atomicAdd( det , logf(x[index]) );
  }
}

///https://math.stackexchange.com/questions/3158303/using-cholesky-decomposition-to-compute-covariance-matrix-determinant
float BlockGpr::logDetCholBlock(CudaMat<float> cholBlock){
  float *det;
  cudaMallocManaged(&det, 4);
  *det = 0;

  dim3 threads_per_block (1024, 1, 1);
  dim3 number_of_blocks ((cholBlock.rows() / threads_per_block.x) + 1, 1, 1);
  logDetCholKernel <<< number_of_blocks, threads_per_block >>>
                    (cholBlock.getRawDevPtr(),
                     cholBlock.ld(), cholBlock.cols(),
                     det);
  cudaStreamSynchronize(0);

  float out = *det;
  out = 2*out;
  return  out;
}

float BlockGpr::logDetChol(){
  float logDet=0;
  for(size_t i=0;i<_chol.rows();i++) {
    logDet += logDetCholBlock(_chol.getBlock(i,i));
  }
  return logDet;
}

void BlockGpr::computeAlpha(){
  // Compute alpha where...
  // alpha = K^-1 y
  alpha= _yTrain.deepCopy();
  ss::choleskySolve(_chol,alpha);
}

float BlockGpr::lmlBetaTerm(){
  CudaMat<float> beta(1,1,dev);  //beta = y^T K^-1 y
  for(size_t i=0;i<_yTrain.rows();i++) {
    CudaMat<float> product=mult(_yTrain.getBlock(i,0),alpha.getBlock(i,0),CUBLAS_OP_T,CUBLAS_OP_N);
    beta += product;
//    product.dev2host();
//    product.printHost();
  }
  beta.dev2host();
  return beta(0,0);
}

float BlockGpr::lml(){
  computeAlpha();
  float logDet = logDetChol();
  float beta = lmlBetaTerm();
  size_t n = _chol.rows() * _chol.getBlockParam().rows;
  return -beta/2  - logDet/2 - n/2 * log(2*M_PI);
}

float BlockGpr::derivativeLML(size_t hp_index){
  computeAlpha();
  CudaBlockMat<float> alpha_squared_minus_k_inv = ss::mult(alpha,alpha,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_TRANSPOSE);
  CudaBlockMat<float> k_inverse = ss::blockMatIdentity(_chol.rows(),_chol.getBlockParam().rows);
  ss::choleskySolveMat(_chol,k_inverse);
  ss::add(alpha_squared_minus_k_inv,k_inverse,-1);
  k_inverse.reset();
  CudaBlockMat<float> d_k;
  d_k.setBlockParam(_chol.getBlockParam());
  for(size_t row = 0; row<_chol.rows() ; row++){
    for(size_t col = 0; col<_chol.cols() ; col++){
      CudaMat<float> partial = params.regression->kernel->partialDerivative(_xTrain.getBlock(row),_xTrain.getBlock(col),hp_index);
      if(partial.nnzCached()>0){
        d_k.pushBackHost(
              partial,
              row,
              col);
      }
    }
  }
  auto product = ss::mult(alpha_squared_minus_k_inv,d_k);
  float out = ss::trace(product);
  out = out*0.5f;
  return out;
}

gpr::Prediction BlockGpr::predict(CudaMat<float> predGrid){
    gpr::Prediction out;
    out.points=predGrid;
    out.mu.reset(predGrid.rows(),1);
    out.mu.initDev();

    out.sigma.reset(predGrid.rows(),1);
    out.sigma.initDev();

    PerfTimer compute_alpha;
    computeAlpha();
    //std::cout << "    compute_alpha: " << compute_alpha.elapsed() << std::endl;
    // Compute mu and sigma


    PerfTimer predict_timer;
    PerfTimer cov_timer;
    cov_timer.pause();
    PerfTimer mult_timer;
    mult_timer.pause();
    CudaBlockMat<float> kp_x;
    denseParam_t kp_x_param(predGrid.rows(), _chol.getBlockParam().cols);
    kp_x.setBlockParam(kp_x_param);

    // kInv_kp_x starts out as the trasnpose of kp_x;
    CudaBlockMat<float> kInv_kp_x;
    denseParam_t kInv_kp_x_param(_chol.getBlockParam().cols, predGrid.rows());
    kInv_kp_x.setBlockParam(kInv_kp_x_param);

    for(size_t i=0;i<_xTrain.rows();i++){
        CudaMat<float> predBlock;

        cov_timer.resume();
        CudaMat<float> kp_x_block = getKernel()->genCovariance(predGrid,_xTrain.getBlock(i,0));//genCovariance(predGrid,_xTrain.getBlock(i,0));
        kp_x.pushBackHost(kp_x_block,0,i);
        CudaMat<float> kInv_kp_x_block = getKernel()->genCovariance(_xTrain.getBlock(i,0),predGrid);  // we need this to be the transpose of kp_x_block
        kInv_kp_x.pushBackHost(kInv_kp_x_block,i,0);
        cov_timer.pause();

        mult_timer.resume();
        predBlock = kp_x_block * alpha.getBlock(i,0);

        out.mu+=predBlock;
        mult_timer.pause();
    }

//    std::cout << "    predict_timer: " << predict_timer.elapsed() << std::endl;
//    std::cout << "        cov_timer: " << cov_timer.elapsed() << std::endl;
//    std::cout << "        mult_timer: " << mult_timer.elapsed() << std::endl;
    CudaMat<float> covariance(predGrid.getDenseParam(),dev);

    PerfTimer chol_timer;
    ss::choleskySolve(_chol,kInv_kp_x);
    covariance = getKernel()->genCovariance(predGrid,predGrid);
    CudaMat<float> product = ss::singleRowMult(kp_x,kInv_kp_x,0,0);
    covariance -= product;
//    std::cout << "    compute_var: " << compute_var.elapsed() << std::endl;

    out.sigma = ss::getDiagonal(covariance);


    // Compute LML
    //out.LML = lml();
    // -1/2 beta - 1/2 log|K| - n/2 log(2pi)

    prediction=out;
    return prediction;
}



gpr::Prediction BlockGpr::predict(float x1Min, float x1Max, float x2Min, float x2Max, size_t x1Div, size_t x2Div){
    CudaMat<float> predGrid = genPredBlock( x1Min,  x1Max, x2Min,  x2Max,  x1Div,  x2Div);
    predGrid.dev2host();
//    std::cout<<"pred grid"<<std::endl;
//    predGrid.printHost();
    return predict(predGrid);
}

