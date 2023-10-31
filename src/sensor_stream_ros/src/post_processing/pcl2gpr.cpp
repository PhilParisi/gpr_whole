#include "../include/sensor_stream_ros/multires_pcl_gpr.h"

int main(int argc, char **argv){
    PredParams corseParam;
    corseParam.tileDim = 8;
    corseParam.trainingDim = 10;
    corseParam.divx = 80;
    corseParam.divy = 80;

    corseParam.hyperParm.reset(2,1);
    corseParam.hyperParm.initHost();
    corseParam.hyperParm(0,0)= 3.0f;  //  length scale
    corseParam.hyperParm(1,0)= 1.0f;    //  alpha
    corseParam.hyperParm.host2dev();
    corseParam.blockSize = 1200;
    corseParam.sigma2e = 1.4f;
    corseParam.randomSampleFactor = 0.05f;
    corseParam.maxPointsPerTile = 40000;

    PredParams fineParam;
    fineParam.tileDim = 8;
    fineParam.trainingDim = 3;
    fineParam.divx = 30;
    fineParam.divy = 30;

    fineParam.hyperParm.reset(2,1);
    fineParam.hyperParm.initHost();
    fineParam.hyperParm(0,0)= 1.20606f;  //  length scale
    fineParam.hyperParm(1,0)= 1.0f;    //  alpha
    fineParam.hyperParm.host2dev();
    fineParam.blockSize = 1200;
    fineParam.sigma2e = 1.0f;
    fineParam.randomSampleFactor = -1.0f;
    fineParam.maxPointsPerTile = 30000;
    MultiresPclGpr predictor;

    //predictor.predictTile(73,176);
    predictor.readCloud(argv[1]);
    predictor.setOutputDir(argv[2]);
    predictor.setParams(corseParam,fineParam);
    predictor.autoTile();


//    pclPredictor finePredictor;
//    finePredictor.readCloud("/home/kris/Dropbox/Ubuntu/Data/wiggles_bank/slice.pcd");
//    finePredictor.setPredParams(fineParam);

//    pclPredictor corsePredictor;
//    corsePredictor.readCloud("/home/kris/Dropbox/Ubuntu/Data/wiggles_bank/slice.pcd");
//    corsePredictor.setPredParams(corseParam);




//    pclPredictor finePredictor;
//    finePredictor.readCloud("/home/kris/Dropbox/Ubuntu/Data/wiggles_bank/slice.pcd");
//    finePredictor.setPredParams(fineParam);
//    //finePredictor.filter2Tile(50.0,200.0);
//    //finePredictor.subsampleInputCloud(0.1);

//    // prepare prediction grid
//    CudaMat<float> predGrid(finePredictor.getCloud()->size(),2,host);
//    for (size_t pointIndex = 0; pointIndex<finePredictor.getCloud()->size(); pointIndex++) {
//        predGrid(pointIndex,0) = finePredictor.getCloud()->points[pointIndex].x;
//        predGrid(pointIndex,1) = finePredictor.getCloud()->points[pointIndex].y;
//    }
//    predGrid.host2dev();

//    pclPredictor corsePredictor;
//    corsePredictor.readCloud("/data/testing/SensorStream/wiggles_bank/slice_subsampled_tile.pcd");
//    corsePredictor.setPredParams(corseParam);
//    corsePredictor.predict(predGrid);

//    corsePredictor.savePredCloud("/data/testing/SensorStream/wiggles_bank/test.pcd");

//    for (size_t pointIndex = 0; pointIndex<finePredictor.getCloud()->size(); pointIndex++) {
//       finePredictor.getCloud()->points[pointIndex].z -= corsePredictor.getPredCloud()->points[pointIndex].z;
//    }

//    pcl::io::savePCDFileASCII("/data/testing/SensorStream/wiggles_bank/training.pcd",*finePredictor.getCloud());

//    finePredictor.predict(corsePredictor.minPt.x,corsePredictor.maxPt.x,corsePredictor.minPt.y,corsePredictor.maxPt.y,400,400);
//    finePredictor.savePredCloud("/data/testing/SensorStream/wiggles_bank/pred_flat.pcd");

//    corsePredictor.predict(finePredictor.getGpr()->prediction.points);

//    for (size_t pointIndex = 0; pointIndex<finePredictor.getPredCloud()->size(); pointIndex++) {
//       finePredictor.getPredCloud()->points[pointIndex].z += corsePredictor.getPredCloud()->points[pointIndex].z;
//       finePredictor.getPredCloud()->points[pointIndex].intensity += corsePredictor.getPredCloud()->points[pointIndex].intensity*.01;
//    }

//    finePredictor.savePredCloud("/data/testing/SensorStream/wiggles_bank/pred.pcd");

    return 0;
}
