#ifndef CSV2GPR_WORKER_H
#define CSV2GPR_WORKER_H

#include <QObject>
#include <QFile>
#include <QStringList>
#include <QDebug>
#include <QVector>
#include <sensor_stream/blockgpr.h>
#include <sensor_stream/gpr/sqexpsparse2d.h>
#include <sensor_stream_ros/third_party/qcustomplot.h>

class CSV2GPRWorker : public QObject
{
  Q_OBJECT
public:
  explicit CSV2GPRWorker(QObject *parent = nullptr);
  //void setParams(gpr::GprParams params){params_=params;}
  gpr::GprParams getParams(){return params_;}
  BlockGpr::Ptr getGpr(){return gpr_;}

signals:
  void predict1dFinished(gpr::Prediction pred);
  void hpChanged();
  void lmlComplete(double min, double max);
  void dataLoaded(int size);

public slots:
  void readFile(QString fname);
  void predict1d(float x_min, float x_max, size_t divisions, float y_val);
  void genLML(QCustomPlot *customPlot, QCPColorMap *colorMap, double x_min, double x_max, double y_min, double y_max, size_t nx, size_t ny);
  void setBlockSize(int block_size);

private:
  void data2gpr();
  BlockGpr::Ptr gpr_;
  gpr::GprParams params_;
  gpr::Prediction prediction_;
  std::vector<float> input_x, input_y;
  std::vector<float> output_vect;
  int block_size_;
};

#endif // CSV2GPR_WORKER_H
