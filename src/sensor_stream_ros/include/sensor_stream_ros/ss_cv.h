#ifndef SS_CV_H
#define SS_CV_H

#include <opencv2/opencv.hpp>
#include <sensor_stream/blockgpr.h>
namespace ss{namespace viz{
  cv::Mat sparsity2image(CudaBlockMat<float> inputMat, float multiplier=1.0f);
}}

#endif // SS_CV_H
