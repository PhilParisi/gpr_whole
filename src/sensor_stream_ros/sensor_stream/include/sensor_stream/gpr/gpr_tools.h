#ifndef GPR_TOOLS_H
#define GPR_TOOLS_H

#include "../../sensor_stream/include/sensor_stream/cudamat.h"

CudaMat<float> estimateLikelihood(CudaMat<float> mu_est,CudaMat<float> mu_obs, CudaMat<float> var_est, CudaMat<float> var_obs);

#endif // GPR_TOOLS_H
