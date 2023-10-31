#include "ss_cv.h"

namespace ss{namespace viz{

cv::Mat sparsity2image(CudaBlockMat<float> inputMat, float multiplier){
  CudaMat<int> sparsityMat = inputMat.getSparsityMatrix();
  cv::Mat out(cv::Size(sparsityMat.rows(), sparsityMat.cols()), CV_8UC3);
  cv::Size outsize(sparsityMat.rows()*multiplier,sparsityMat.cols()*multiplier);
  for(size_t i=0; i< sparsityMat.rows() ; i++){
    for (size_t j=0 ; j < sparsityMat.cols() ; j++) {
      if(sparsityMat(i,j)>0){
        int nnz = sparsityMat(i,j);
        int size = inputMat.getBlockParam().cols*inputMat.getBlockParam().rows;
        uint8_t val = nnz*255/size;
        out.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,val);
      }
      else if(sparsityMat(i,j)==0){
        out.at<cv::Vec3b>(i,j) = cv::Vec3b(255,0,0);
      }
      else {
        out.at<cv::Vec3b>(i,j) = cv::Vec3b(132, 189, 153);
      }
    }
  }
  cv::resize(out, out,outsize,0,0);
  return out;
}


}}
