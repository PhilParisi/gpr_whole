#include "multires_pcl_gpr.h"

MultiresPclGpr::MultiresPclGpr()
{

}

void MultiresPclGpr::setParams(PredParams corse, PredParams fine){
  finePredictor.setPredParams(fine);
  corsePredictor.setPredParams(corse);
  corsePredictor.subsampleInputCloud(corse.randomSampleFactor);
}

void MultiresPclGpr::predictTile(float tileCenterX, float tileCenterY){
  finePredictor.filter2Tile(tileCenterX,tileCenterY);
  corsePredictor.filter2Tile(tileCenterX,tileCenterY);

  if(finePredictor.getFilteredCloud()->size()<1){
    return;
  }

  CudaMat<float> predGrid(finePredictor.getFilteredCloud()->size(),2,host);

  float avg_z=0;
  for (size_t pointIndex = 0; pointIndex<corsePredictor.getFilteredCloud()->size(); pointIndex++) {
     avg_z += corsePredictor.getFilteredCloud()->points[pointIndex].z;
  }
  avg_z = avg_z/corsePredictor.getFilteredCloud()->size();

  for (size_t pointIndex = 0; pointIndex<corsePredictor.getFilteredCloud()->size(); pointIndex++) {
    corsePredictor.getFilteredCloud()->points[pointIndex].z -= avg_z;
  }

  // set the pred grid as the x,y positions of the input cloud
  for (size_t pointIndex = 0; pointIndex<finePredictor.getFilteredCloud()->size(); pointIndex++) {
      predGrid(pointIndex,ss_x) = finePredictor.getFilteredCloud()->points[pointIndex].x;
      predGrid(pointIndex,ss_y) = finePredictor.getFilteredCloud()->points[pointIndex].y;
  }
  predGrid.host2dev();


  corsePredictor.predict(predGrid);
  add(corsePredictor.getGpr()->prediction.mu,avg_z);
  corsePredictor.generateCloud();

  std::string fname = output_dir+"corse_pred_"+std::to_string(tileCenterX)+"_"+std::to_string(tileCenterY)+".pcd";
  corsePredictor.savePredCloud(fname);


  for (size_t pointIndex = 0; pointIndex<corsePredictor.getPredCloud()->size(); pointIndex++) {
     finePredictor.getFilteredCloud()->points[pointIndex].z -= corsePredictor.getPredCloud()->points[pointIndex].z;
  }

  finePredictor.predictTile(tileCenterX,tileCenterY);

  fname = output_dir+"fine_pred_grid_"+std::to_string(tileCenterX)+"_"+std::to_string(tileCenterY)+".pcd";
  finePredictor.savePredCloud(fname);

  corsePredictor.predict(finePredictor.getGpr()->prediction.points);
  add(corsePredictor.getGpr()->prediction.mu,avg_z);
  corsePredictor.generateCloud();

  fname = output_dir+"corse_pred_grid_"+std::to_string(tileCenterX)+"_"+std::to_string(tileCenterY)+".pcd";
  corsePredictor.savePredCloud(fname);

  for (size_t pointIndex = 0; pointIndex<finePredictor.getPredCloud()->size(); pointIndex++) {
     finePredictor.getPredCloud()->points[pointIndex].z += corsePredictor.getPredCloud()->points[pointIndex].z;
     finePredictor.getPredCloud()->points[pointIndex].intensity += corsePredictor.getPredCloud()->points[pointIndex].intensity*.01;
  }

  //finePredictor.savePredCloud("/home/kris/Dropbox/Ubuntu/Data/wiggles_bank/output/multires_pred.pcd");

}

void MultiresPclGpr::autoTile(){
  size_t tileno=0;
  for (float i = finePredictor.minPt.x;i<finePredictor.maxPt.x;i+=finePredictor.getPredParams().tileDim) {
    for (float j = finePredictor.minPt.y;j<finePredictor.maxPt.y;j+=finePredictor.getPredParams().tileDim) {
      predictTile(i,j);

      if(finePredictor.getPredCloud() && finePredictor.getPredCloud()->size()>0){
        std::string fname = output_dir+"multires_pred"+std::to_string(tileno)+".pcd";
        //pcl::io::savePCDFileASCII(fname,*finePredictor.getFilteredCloud());
        finePredictor.savePredCloud(fname);
        tileno++;
      }
    }
  }
}
