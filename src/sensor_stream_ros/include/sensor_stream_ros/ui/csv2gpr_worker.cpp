#include "csv2gpr_worker.h"

CSV2GPRWorker::CSV2GPRWorker(QObject *parent) : QObject(parent)
{
  params_.regression.reset(new gpr::RegressionParams);
  params_.prediction.reset(new gpr::PredParams);
  params_.regression->kernel.reset(new ss::kernels::SqExpSparse2d);
  params_.regression->kernel->hyperparam("length_scale")  = 0.6f;
  params_.regression->kernel->hyperparam("process_noise") = 1.0f;
  params_.regression->kernel->hyperparam2dev();

  params_.regression->sensor_var = powf(0.2f,2);
  params_.regression->nnzThresh=1e-16f;

  block_size_=400;

}

void CSV2GPRWorker::readFile(QString fname){


  QFile file(fname);
  if (!file.open(QIODevice::ReadOnly)) {
      qDebug() << file.errorString();
      return;
  }

  input_x.clear();
  input_y.clear();
  output_vect.clear();

  while (!file.atEnd()) {
      QByteArray line = file.readLine();
      auto list = line.split(',');
      input_x.push_back( list[0].trimmed().toFloat() );
      input_y.push_back( list[1].trimmed().toFloat() );
      output_vect.push_back( list[2].trimmed().toFloat());
      //wordList.append(line.split(','));
  }
  emit dataLoaded(input_x.size());

}

void CSV2GPRWorker::data2gpr(){
  gpr_.reset(new BlockGpr);
  gpr_->params=params_;
  size_t i = 0;
  while( i<input_x.size() ){
    CudaMat<float> input(block_size_,2,host);
    CudaMat<float> output(block_size_,1,host);
    for(size_t block_idx = 0; block_idx<block_size_ ; block_idx++){
      if(i<input_x.size()){
        input.x(block_idx)=input_x[i];
        input.y(block_idx)=input_y[i];
        output(block_idx) =output_vect[i];
      }
      i++;
    }
    input.host2dev();
    output.host2dev();
    gpr_->addTrainingData(input,output);
  }
}

void CSV2GPRWorker::predict1d(float x_min, float x_max, size_t divisions, float y_val){
  data2gpr();
  prediction_ = gpr_->predict(x_min,x_max,y_val,y_val,divisions,1);
  //gpr_->getCholeskyFactor().blocks2host();
  //gpr_->getCholeskyFactor().printHostValues();
  prediction_.dev2host();
  //prediction_.mu.printHost();
  emit(predict1dFinished(prediction_));

}

void CSV2GPRWorker::genLML(QCustomPlot * customPlot,  QCPColorMap *colorMap, double x_min, double x_max, double y_min, double y_max, size_t nx, size_t ny){
  double min = INFINITY;
  double max = -INFINITY;

  customPlot->clearGraphs();

  colorMap->data()->setSize(nx, ny);
  customPlot->xAxis->setScaleType(QCPAxis::stLogarithmic);
  customPlot->yAxis->setScaleType(QCPAxis::stLogarithmic);
  QSharedPointer<QCPAxisTickerLog> logTicker(new QCPAxisTickerLog);
  customPlot->yAxis->setTicker(logTicker);
  customPlot->yAxis2->setTicker(logTicker);
  customPlot->xAxis->setTicker(logTicker);
  customPlot->xAxis2->setTicker(logTicker);
  //customPlot->xAxis2->setScaleType(QCPAxis::stLogarithmic);
  //customPlot->yAxis2->setScaleType(QCPAxis::stLogarithmic);
  customPlot->xAxis->setNumberFormat("eb"); // e = exponential, b = beautiful decimal powers
  customPlot->xAxis->setNumberPrecision(0); // makes sure "1*10^4" is displayed only as "10^4"
  customPlot->yAxis->setNumberFormat("eb"); // e = exponential, b = beautiful decimal powers
  customPlot->yAxis->setNumberPrecision(0); // makes sure "1*10^4" is displayed only as "10^4"
  //colorMap->setDataScaleType(QCPAxis::stLogarithmic);
  colorMap->data()->setRange(QCPRange(x_min, x_max), QCPRange(y_min, y_max));

  double log_step_x = log10(x_max)-log10(x_min);
  double log_step_y = log10(y_max)-log10(y_min);
//  for (int x_index=0; x_index<nx; ++x_index){
//    for (int y_index=0; y_index<ny; ++y_index){
//      colorMap->data()->setCell(x, y, qCos(x/10.0)+qSin(y/10.0));
//    }
//  }


  double x, y, z;
  for (int xIndex=0; xIndex<nx; ++xIndex)
  {
    double log_x=log10(x_min)+log_step_x*xIndex/nx;
    x = pow(10,log_x);
    for (int yIndex=0; yIndex<ny; ++yIndex)
    {
      //colorMap->data()->cellToCoord(xIndex, yIndex, &x, &y);

      double log_y=log10(y_min)+log_step_y*yIndex/ny;
      y = pow(10,log_y);
      params_.regression->kernel->hyperparam(0)=x;//pow(10,x);
      params_.regression->kernel->hyperparam(1)=y;//pow(10,y);
      params_.regression->kernel->hyperparam2dev();
      data2gpr();
      float lml = gpr_->lml();
      if(lml<min){
        min = lml;
      }
      if(lml>max){
        max = lml;
      }
      colorMap->data()->setCell(xIndex, yIndex, lml);
    }
  }

  customPlot->xAxis->setLabel(QString::fromStdString(gpr_->params.regression->kernel->getHyperparamKey(0)));
  customPlot->yAxis->setLabel(QString::fromStdString(gpr_->params.regression->kernel->getHyperparamKey(1)));

  // add a color scale:
  QCPColorScale *colorScale = new QCPColorScale(customPlot);
  customPlot->plotLayout()->addElement(0, 1, colorScale); // add it to the right of the main axis rect
  colorScale->setType(QCPAxis::atRight); // scale shall be vertical bar with tick/axis labels right (actually atRight is already the default)
  colorMap->setColorScale(colorScale); // associate the color map with the color scale
  colorScale->axis()->setLabel("LML");

  // set the color gradient of the color map to one of the presets:
  colorMap->setGradient(QCPColorGradient::gpJet);
  // we could have also created a QCPColorGradient instance and added own colors to
  // the gradient, see the documentation of QCPColorGradient for what's possible.

  // rescale the data dimension (color) such that all data points lie in the span visualized by the color gradient:
  colorMap->rescaleDataRange();

  // make sure the axis rect and color scale synchronize their bottom and top margins (so they line up):
  QCPMarginGroup *marginGroup = new QCPMarginGroup(customPlot);
  customPlot->axisRect()->setMarginGroup(QCP::msBottom|QCP::msTop, marginGroup);
  colorScale->setMarginGroup(QCP::msBottom|QCP::msTop, marginGroup);

  // rescale the key (x) and value (y) axes so the whole color map is visible:
  customPlot->rescaleAxes();
  customPlot->replot();
  emit lmlComplete(min,max);

}

void CSV2GPRWorker::setBlockSize(int block_size){
  block_size_=block_size;
}
